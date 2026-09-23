# 桌面端接入 v2 — 改造清单

> 配套文档：`docs/architecture-v2.md`（服务端设计）。原则：**`ZSession` 是唯一切换点**，v2 API 客户端替换掉旧客户端（v1 服务端已删除，客户端不再保留双协议）。因为没有兼容层，桌面端与服务端必须**同步发布**：上线前确认两者版本匹配（登录时用 `/api/v2/health` 校验）。

## 0. 影响面总览

现有分层已经把改动面收窄了：`ZSession`（`zlabel/utils/session.py`）是 storage/inference 的唯一组装点，`RemoteStorage`/`RemoteInference`（`zlabel/utils/backend.py`）已经抽象了所有服务端调用。因此：

| 层 | 文件 | 改动 |
|---|---|---|
| API 客户端 | `zlabel/utils/api_helper.py` | 重写为 v2 协议的 `ZLServerApiClient`（方法名尽量保持不变，减少上层改动）；旧的 `{message,data}` 信封逻辑删除 |
| 组装 | `zlabel/utils/session.py` | 直接组装 v2 客户端；暴露 `claim/release/submit/review/versions` 等新能力，并把登录时探测到的 `capabilities` 透给 UI |
| Storage/Inference | `zlabel/utils/backend.py` | `RemoteStorage` 补领取/状态/版本方法；`RemoteInference` 去掉"当前任务"假设 |
| 模型 | `zlabel/utils/project.py` | `Task` 增 `state/claimed_by/lease_expires_at/version/group/day(服务端)`；`User` 增 `role` |
| 设置 | `zlabel/widgets/zsettings.py` | 租约心跳/自动释放、服务端能力缓存、服务端版本 |
| UI | `zlabel/widgets/*` | 领取冲突对话框、复核动作、版本历史、任务状态徽标、标签编辑（按角色） |
| i18n/文档 | `i18n/*.ts`、`AGENTS.md`、`CHANGELOG.md` | 新字符串与新行为记录 |

## 1. 分阶段清单

> 进度：**C1–C12 全部完成**（协议层 C1–C3+C7、协作层 C4–C6、第三批 C8–C12）。
> 服务端侧配套改动：`ProjectOut.id`、`state` 支持逗号列表、`order=random`、`scan` 不要求项目已存在、
> worker 侧 crop 提示词平移修复、心跳只续活租约（提交后需重新领取）、读写令牌分离
> （读=用户令牌、写=服务账号）。
> C8 任务图像磁盘缓存（`~/.zlabel/cache/images` + ETag/If-None-Match，304 走本地）、
> C9 标签表编辑（reviewer+，逐条提交）、C10 新建服务端项目（目录+标记文件+自动扫描）、
> C11 zh_CN 全量翻译（375 条，0 unfinished）、C12 端到端覆盖：`tests/v2/test_client_contract.py`
> 含"两人抢同一任务 → 租约过期 → 接管 → 前者保存被拒（带持有者名）→ reviewer 强制接管"，
> 以及领取→保存→提交→复核→退回→重开与版本历史全链路。
> 剩余：仅 M5 验收（DoD 逐条走查 + 真实数据集/CUDA 环境试跑）。


### C1 版本与能力探测（阻塞项）
- 登录成功后 `GET /api/v2/health`：校验 `version >= 2.0`，并把 `capabilities`（`tasks.claim`/`tasks.review`/`annotations.versions`/`labels.write`/`predict.stateless`）存入 session。
- 服务器版本不匹配 → 明确提示「服务端版本过旧，请同步升级」，不进入标注界面（避免半可用状态）。
- `session.capabilities` 供 UI 决定是否显示领取/复核/版本入口；服务端降级（如推理不可用）只灰掉相关动作，不影响标注与保存。

### C2 认证与会话
- 登录响应新增 `user{name,role}` 与 `expires_at`；写入 `settings.username`/状态栏（显示角色：标注员/复核员/管理员）。
- 新增 `logout()`；会话失效（401 `session_stale`）→ 自动重登一次，失败则弹"请重新登录"。
- token 语义变化：客户端持有的是**服务端签发的 session token**，仅内存持有；401（含 `session_stale`）时重新登录。

### C3 任务拉取（去掉文件名硬猜）
- 改用 `GET /api/v2/projects/{p}/tasks?state=&claim=&mine=&limit=&order=sequence`；直接用服务端 `group/day`。
- 删除 `MainWindow._assign_remote_group`（分组改由服务端给出）。
- Fetch 过滤：`FetchType` 由 `FINISHED/UNFINISHED/ALL` 扩展为按 `state`（`draft/submitted/approved/rejected/all/mine`），设置项与文件 dock 的下拉同步更新。
- 时间轴/拷贝上一任务改走 `GET /api/v2/projects/{p}/groups`（服务端给整组任务，不再受 `num` 截断影响）。

### C4 领取与租约（v2 新增协作核心）
- `try_set_image` 成功后 `claim(anno_id)`；返回 409 → 新对话框：`{claimed_by, lease_expires_at}` + 三个按钮「等待/查看其他任务/强制接管（需 reviewer+）」。
- 心跳：`QTimer` 每 10 分钟 `heartbeat(anno_id)`（仅当前任务持有者，且非只读）；切换任务时 `release` 上一任务（可由设置 `auto_release_on_switch` 控制）。
- 保存成功后自动续租（服务端已幂等处理），状态徽标刷新。

### C5 保存与版本冲突
- `save_zlabel` 增 `base_version`（来自最近一次读取）；服务端 409 → 复用现有"标注冲突"弹窗，展示服务端 `version/updated_by/updated_at`，按钮「重新加载服务端版本 / 强制覆盖（reviewer+）」。
- 新增「版本历史」对话框：列表（版本号/作者/时间/标签数）+ 只读预览；reviewer 可「回滚到该版本」（= 以旧内容 + 新 `base_version` 保存）。

### C6 提交与复核
- 新增动作（受 `capabilities` 与角色控制）：`Submit for review`（annotator 提交当前任务；可"提交本组"批量）、`Approve`/`Reject…`（reviewer/admin，Reject 弹输入框填 note）。
- 任务状态在文件列表与信息面板显示徽标（草稿/待复核/已通过/已退回），退回原因在提示条/信息面板可见。
- 文件 dock 的 `Fetch` 在 v2 下重命名为「Scan」（触发服务端重扫）。

### C7 推理调用（修掉错图风险）
- 改调 `POST /api/v2/projects/{p}/predict`：请求带 `image_sha256`（客户端算）或直接带图；服务端按图缓存 embedding，**不再依赖"服务端当前图像"**。
- 删除 `RemoteInference.local_images` 的分支差异：v2 统一"带图或带图引用"，`server`/`local` 图源只影响取图方式。
- 503 `inference_unavailable` → 提示"推理服务不可用，可继续手动标注"，不阻塞标注与保存。

### C8 图像获取
- 改调 `GET /api/v2/projects/{p}/images/{rel_path}`（ETag/`If-None-Match`）；v2.1 可加磁盘缓存（`~/.zlabel/cache/<sha>`）。
- 相对路径口径统一：**客户端一律传项目内相对路径 + project**；不再依赖服务端返回绝对路径（旧协议把上游绝对路径塞在 `filename` 里）。

### C9 标签
- 从 `GET /api/v2/projects/{p}/labels` 拉取（含颜色/排序/归档），替换现在"从已上传标注里合并名字"的做法。
- reviewer/admin 可在标签面板编辑并 `POST/PATCH/DELETE` 推送；annotator 只读（UI 置灰并提示）。

### C10 项目与进度
- `Add Server Project...` 改读 `GET /api/v2/projects`（含我的进度）；v2 下可新增「新建项目」（admin/reviewer）。
- 状态栏进度改为 `{draft, submitted, approved, rejected} / total`；"我的"统计来自 `GET /api/v2/projects/{p}/progress?by_user=true`。

### C11 i18n 与文档
- 新增字符串（领取/复核/版本/角色/状态）走 `uv run zlabel-uic`（lupdate）与 `uv run zlabel-translate`。
- 更新 `AGENTS.md`（session 鉴权、领取/复核、v2 端点、版本匹配要求）与 `CHANGELOG.md`。

### C12 测试
- 单测：`ZLServerApiHelperV2`（fake HTTP：登录/领取冲突/409 版本冲突/503 推理）——用 `httpx`/`responses` 或注入 session stub。
- GUI：`tests/gui/` 增补领取冲突对话框、复核动作、状态徽标用例；`tests/gui/conftest.py` 的 `main_window` 需要能注入带 capabilities 的假 session。
- 端到端（可选）：`tests/e2e/`（标记 `slow`）对真服务端跑一遍"两人抢同一任务 → 复核退回"。

## 2. 新增/变更的设置项

| 设置项 | 默认 | 说明 |
|---|---|---|
| `auto_release_on_switch` | `true` | 切换任务时主动释放租约 |
| `lease_heartbeat_minutes` | `10` | 心跳间隔（服务端 TTL 30 分钟） |
| `claim_conflict_action` | `ask` | 撞车时：`ask`/`skip`/`wait` |
| `submit_scope` | `task` | 提交范围：`task`/`group`（序列整组） |

## 3. 客户端数据模型变更

```python
class Task(BaseModel):
    ...
    state: Literal["draft", "submitted", "approved", "rejected"] = "draft"
    version: int = 0  # 服务端标注版本（乐观锁 base_version）
    claimed_by: str = ""  # 当前持有者
    lease_expires_at: datetime | None = None
    server_group: str = ""  # v2 服务端解析的分组（优先于本地 group 猜测）
    review_note: str = ""


class User(BaseModel):
    ...
    role: Literal["annotator", "reviewer", "admin"] = "annotator"
```

## 4. 发布与回退

- 没有 `/api/v1` 兼容层：服务端与桌面端必须**同批次发布**，建议先发服务端（旧客户端连上会得到明确的版本不匹配错误，而不是静默行为异常）。
- 实施顺序（同一发布批次内）：C1→C2→C3→C7（协议层，风险最低）→C5→C4→C6→C8→C9→C10。
- 回退：客户端与服务端一起退回上一版（v2 内部的版本）；数据面不变（v2 一直用同一个新库），因此回退不需要数据迁移。
- 数据面：v2 用全新数据库；旧库（`zlabel_server.db`）与 `/api/v1` 一起废弃，不再参与任何流程。

## 5. 验收（客户端侧）

1. 打开一个任务被他人持有 → 弹出持有者与剩余租约，不丢当前工作。
2. 保存后他人以旧版本保存 → 服务端 409，客户端展示冲突详情并可重新加载。
3. annotator 看不到复核入口；reviewer 可退回并填写原因，退回原因在客户端可见。
4. 推理服务不可用时，标注/保存/复核全部照常，仅提示推理不可用。
5. 服务端版本不匹配时给出明确提示并阻止进入标注流程；服务端能力缺失时对应入口隐藏/灰置，不出现半可用状态。

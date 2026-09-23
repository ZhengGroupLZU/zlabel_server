# ZLabel Server v2 — 架构设计

> 状态：已实现并上线（见 §12 M0–M3、M5）。**存储与身份部分在 P6 后已改为自建**：服务器掌管
> `ZLSERVER_STORAGE_ROOT` 目录树、账号存在自己的 `users` 表（scrypt），OpenList 的 SDK/适配器/
> 配置开关已全部删除（见 `plan-selfhosted-storage.md`）。§2 D1、§3/§4/§6/§7 中遗留的 OpenList
> 字样是历史设计记录，不再是实现。v1（分支 `onnx`，提交 `fc04ef4`）仅作参考留档。

## 1. 目标与非目标

**目标**：把标注服务端从"单用户脚本式后端"升级为**多用户生产系统**：任务领取与租约、提交/复核状态机、角色权限与审计、按人统计、服务端标签管理、推理与 API 解耦（不再有"全局当前帧"），并且 HTTP 契约有版本、可灰度、可回滚。

**非目标（v2 不做）**：不做通用 Web 前端（只保留 `/admin` 运维后台）；不改桌面端的本地（离线）模式；不引入 Postgres/Redis 等外部中间件（保持单机可部署，但边界留好）。

> 注："不替换 OpenList 作为文件存储"这一条后来被推翻——P0–P6 把存储/身份换成了自建实现。

## 2. 已确认的四项决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | **服务端自己的账号 + 自己的存储树**：账号是 `users` 表的 scrypt 哈希，登录后签发自己的 session；数据集就是 `ZLSERVER_STORAGE_ROOT` 下的普通目录，标注写在 `<project>/.zlabel/annos/`（与桌面端数据集布局一致）。原设计"OpenList 作身份源 + 文件存储"已在 P6 被本方案取代 | 单机可部署、不依赖外部服务；与桌面端本地数据集可互换；角色/领取/审计能力保留在服务端 |
| D2 | **只保留 `/api/v2`，v1 彻底删除** | 少维护一套适配代码；代价是没有灰度、服务端与桌面端必须一次性同步升级（见 §10、§12） |
| D3 | **推理独立进程 + 队列** | 消除"全局当前帧"错帧 bug；API 重启不影响模型；GPU 排队不占 API worker；可水平扩 GPU |
| D4 | **领取制 + 复核状态机** | 多人协作不撞车（租约自动过期）；复核可退回；进度按状态统计 |

## 3. 目录结构

v1 已删除（代码留档在 `onnx` 分支）。可复用资产先搬出，再删 v1：

```
inference/               # 从 app/ 搬来的推理资产（sam_ort/、worker.py、ztypes.py、
                         # bbox_overlaps.py、debug_save.py、config.py、logging.py）
v2/
  main.py                # FastAPI 装配（include v2 routers + v1 兼容 router）+ lifespan
  core/
    config.py            # pydantic-settings，ZLSERVER_* 前缀
    logging.py           # 结构化日志 + request id
    errors.py            # 统一错误模型 {code, message, detail}
    security.py          # session 签发/校验、角色依赖（require_user / require_role）
  db/
    base.py              # engine/session 工厂
    models.py            # SQLAlchemy 模型（见 §4）
    repositories/        # 每个聚合一个仓库（projects/tasks/annotations/users/labels/audit）
    migrations/          # alembic
  adapters/
    storage.py           # StorageBackend 协议（路径约定 + IO 方法）
    local_disk.py        # 本地目录树实现（原子写、路径穿越防护、usage/delete）
    identity.py          # IdentityProvider + LocalIdentity（scrypt，标准库）
    inference.py         # InferenceClient（HTTP 调推理进程）
  admin/                 # starlette-admin 后台（cookie 会话复用 AuthService，仅 admin 角色）
  services/
    auth_service.py      # 登录、会话、角色引导（首个用户=admin）
    project_service.py   # 项目发现/创建/同步、标签 CRUD
    task_service.py      # 任务列表、分组解析、领取/释放/心跳、提交/复核
    annotation_service.py# 读写标注、版本历史、冲突检测
    stats_service.py     # 进度、按人统计
  api/v2/
    auth.py projects.py tasks.py annotations.py images.py predict.py labels.py health.py
  contracts/             # 与桌面端共享的线格式（转发 inference/ztypes 的枚举与结果类型）
  schemas/               # 请求/响应模型（pydantic v2）
  inference_worker/      # 独立进程的 entrypoint（复用 inference/ 的资产）
docs/
  architecture-v2.md     # 本文
  client-migration-v2.md # 桌面端改造清单
```

## 4. 领域模型与 DB schema

**全新数据库**（`ZLSERVER_DATABASE_URL`，默认 `data/zlabel_server_v2.db`）：
v2 **不迁移、不导入 v1 数据**（旧项目不保留；任务表本来就由扫描存储树重建，标注文件按
`anno_id` 存取，历史标注不丢）。
v1 的 `zlabel_server.db` 留给旧服务端（M4 双跑对比需要两者并存）。
schema 由 **alembic** 管理（`uv run alembic upgrade head`），首次即建全量 v2 表：

```
users            id, identity_id(uniq, "<provider>:<subject>"), password_hash(scrypt), name(uniq, lower),
                 email, role[annotator|reviewer|admin], active, created_at, last_login_at, finished_count
sessions         id, token_hash(uniq), user_id, created_at, expires_at, revoked_at, client_info
project_members  id, project_id, user_id, role, created_at   (uniq(project_id,user_id))
projects         id, name(uniq, = 存储根下的目录名), key(uniq, = 数据集 project.json 的 id),
                 display_name, description, active, timeline(是否支持时间线，默认 true),
                 created_at, updated_at
labels           id, project_id, name, color, sort, archived, created_at   (uniq(project_id,name))
tasks            id, project_id, anno_id(uniq), path(存储根路径), rel_path, group_name, day,
                 state[draft|submitted|approved|rejected], missing,
                 claimed_by, claimed_at, lease_expires_at,
                 submitted_at, reviewed_by, reviewed_at, review_note, updated_at
annotations      id, task_id(uniq), anno_id, version(int), author_id, path, labels_json, hash, created_at, updated_at
instances        id, project_id, number(uniq per project = 文档里的 instance_id), name, note,
                 status, color, archived, created_at, updated_at
instance_results  id, instance_id, task_id, result_id, label    (uniq(task_id,result_id)：一个 result 只属于一个实例)
annotation_versions  id, task_id, version, author_id, path, labels_json, created_at, note   (历史)
link_task_label  (task_id, label_id)      # 保留，v1 兼容读
link_task_user   (task_id, user_id)       # 保留
audit_log        id, ts, user_id, action, target_type, target_id, detail_json
```

要点：

- **`tasks.anno_id` = `sha256("<project key>/<项目内相对路径>")`**：project key 是数据集 `.zlabel/project.json` 的 `"id"`（桌面端 `Project.id`），服务端存在 `projects.key`；改目录名/显示名不会失效。旧数据用 `uv run python -m v2.cli migrate-anno-ids` 一次性改名迁移。
- **标注文件在存储树里**（`<storage_root>/<project>/.zlabel/annos/<anno_id>.zlabel`，布局可用 `ZLSERVER_ANNO_DIR` 覆盖；历史版本在 `annos/_history/<anno_id>/v<n>.zlabel`），DB 只存元数据/版本/哈希 → 与桌面端镜像互通不变。
- `tasks.rel_path` 是**项目内相对路径**（`images/a/D1.png`），`tasks.path` 是存储根下的路径；`get_image` 一律走 `project + rel_path`，修掉 v1 的 `get_image` 忽略 `project` 的口径问题。每个顶层目录都是一个项目（无标记文件）。
- `tasks.group_name/day` 由服务端解析（规则见 §9），客户端不再靠文件名硬猜。
- 领取即租约：`claimed_by/claimed_at/lease_expires_at`；保存成功或提交时自动续租/释放。
- 索引：`tasks(project_id,state)`、`tasks(project_id,group_name,day)`、`tasks(anno_id)`、`sessions(token_hash)`、`audit_log(ts)`。

## 5. 任务状态机

```
                claim                 submit                approve
  (none) ──────────────▶ claimed ──────────────▶ submitted ──────────────▶ approved
    ▲      lease TTL 到期/ release        ▲                 │ reject
    │                                     └─────────────────┘
    └──────────────── reviewer/admin 强制释放 ────────────────┘
```

- `POST /api/v2/tasks/{anno_id}/claim` → 200（我是持有者）/ 409（他人持有，带 `claimed_by/lease_expires_at`）；`force=true` 仅 reviewer/admin。
- 保存标注会自动 `claim + 续租`（幂等）；租约 TTL 默认 30 分钟，客户端每 10 分钟心跳。
- `submit` 把 `draft → submitted`（annotator 即可）；`review` 支持 `approve|reject`（reviewer/admin），`reject` 必须带 note，退回 `draft` 并清空领取。
- 进度：`{total, draft, submitted, approved, rejected}`，可 `by_user=true` 按人统计。

## 6. API v2 契约（摘要）

统一响应：成功直接返回资源对象（**不再有 `{message, data}` 信封**）；错误统一 `{code, message, detail}`，`code` 为机器可读枚举（`unauthorized`/`forbidden`/`not_found`/`conflict`/`lease_conflict`/`validation_error`/`upstream_error`）。

| 方法与路径 | 权限 | 说明 |
|---|---|---|
| `POST /api/v2/auth/login` | — | body `{username,password,client}` → `{token,expires_at,user{id,name,role}}` |
| `GET /api/v2/auth/me` | session | 当前用户 + 角色 + 统计 |
| `POST /api/v2/auth/logout` | session | 撤销当前会话 |
| `GET /api/v2/health` | — | `{db[, storage, inference], version, capabilities[]}`（能力探测用；`deep=true` 才探 storage/inference） |
| `GET /api/v2/projects` | session | 项目列表（含我的进度） |
| `POST /api/v2/projects` | admin/reviewer | 建项目（在存储根下建目录 + DB 行） |
| `PATCH /api/v2/projects/{p}` | admin/reviewer | 描述/显示名/启停/`timeline`（关掉则不解析 group/day，切换时立即重算已有任务） |
| `DELETE /api/v2/projects/{p}?delete_files=` | admin | 删项目（不可逆）：默认连同目录一起删；`delete_files=false` 只注销（磁盘上保留，重扫会重新收录）。子行显式删除（SQLite 不强制 CASCADE），删文件在事务内，失败则整体回滚 |
| `POST /api/v2/projects/{p}/scan` | reviewer/admin | 重扫存储树同步任务（替代 `refresh_tasks`）；**扫描是唯一手动触发的操作**，服务端没有开机/定时扫描 |
| `GET/POST/PATCH/DELETE /api/v2/projects/{p}/labels[/{id}]` | 读 session / 写 reviewer+ | 标签 CRUD（名字、颜色、排序、归档）；标签的 **id 就是顺序号**（0 起，即导出的类别 id），新建默认追加 |
| `GET/POST/PATCH/DELETE /api/v2/projects/{p}/instances[/{number}]` | 读 session / 写 reviewer+ | 项目级实例（文档 `instance_id` 指向它）：状态/名称/备注/颜色/归档；编号**不可改**；`GET /{number}/results` 看它包含的标注，`GET /statuses` 给状态候选 |
| `PUT /api/v2/projects/{p}/labels/order` | reviewer+ | 重排标签（body `{"order": [id...]}`，必须恰好包含全部标签）；服务端把 `sort` 重写为 0..N-1，主键 `Label.id` 不变 |
| `GET /api/v2/projects/{p}/tasks` | session | `state,claim,group,mine,limit,cursor,order=id\|sequence` |
| `POST /api/v2/tasks/{anno_id}/claim\|release\|heartbeat` | session | 领取/释放/续租 |
| `POST /api/v2/tasks/{anno_id}/submit` | 持有者 | 提交复核 |
| `POST /api/v2/tasks/{anno_id}/review` | reviewer/admin | `{decision: approve\|reject, note}` |
| `GET /api/v2/projects/{p}/groups` | session | 序列分组 + 帧列表（时间轴/拷贝上一帧用，服务端给全量，不受分页截断） |
| `GET /api/v2/projects/{p}/my-stats` | session | 我名下各状态的任务数 |
| `POST /api/v2/tasks/{anno_id}/reopen` | reviewer/admin | 已通过/已提交退回草稿（复核者改主意） |
| `GET /api/v2/auth/users` · `PUT /api/v2/auth/users/{id}/role` | admin | 用户列表 / 改角色（改角色会撤销该用户会话） |
| `GET /api/v2/projects/{p}/images/{rel_path:path}` | session | 取图（直接读存储树），支持 `ETag`/`If-None-Match`；**无副作用** |
| `GET /api/v2/images/{sha256}` | session | 取回客户端上传过（内容寻址）的帧 |
| `PUT /api/v2/projects/{p}/images/{rel_path:path}` | session | 本地上传（本地数据集 + 远端推理） |
| `GET /api/v2/projects/{p}/annotations/{anno_id}` | session | 200 + `ETag: v{n}` / 404 = 未标注 |
| `PUT /api/v2/projects/{p}/annotations/{anno_id}` | 持有者 | body 含 `base_version`；200 `{version}` / 409 冲突（`server_version, updated_by, updated_at`）/ `force=true` 仅 reviewer+ |
| `GET /api/v2/projects/{p}/annotations/{anno_id}/versions[/{n}]` | session | 版本历史 / 某一版内容（历史文件存 `zlabel/_history/<anno_id>/v<n>.zlabel`） |
| `POST /api/v2/projects/{p}/predict` | session | 交互/文本推理，**无状态**：multipart `data`(JSON) + 可选 `image`；`data` 里可带 `rel_path`（服务端去存储树取）或 `image_sha256`（取上传缓存） |
| `GET /api/v2/projects/{p}/progress` | session | `{total,draft,submitted,approved,rejected}`，`by_user=true` 时带按人明细 |

错误码约定：`404 not_found`（未标注/未找到，客户端可安全新建）、`409 conflict`（版本冲突或任务被他人领取，`detail.claimed_by` 区分）、`502 upstream_error`（存储/推理进程失败）、`503 inference_unavailable`。

后台管理：`/api/v2/admin/users`（建号/改角色/启停/改密）、`/api/v2/admin/storage`、`/api/v2/admin/files`；项目成员 `/api/v2/projects/{p}/members`；项目实例 `/api/v2/projects/{p}/instances`。Web 后台 `/admin`（Dashboard / Users / Projects / Files / Audit log）把这些写操作重新走服务层，审计与会话吊销与 API 一致；项目范围内的文件、成员、标签、实例、帧都在项目详情页（实例页：编号只读、其余字段一个 Save all 批量提交）。

## 7. 鉴权与会话

1. 客户端 `POST /api/v2/auth/login`（字段与现在一致：用户名/密码）。
2. 服务端用 `LocalIdentity` 校验 scrypt 哈希（`IdentityProvider` 是可替换的接缝）→ 解析/建立 `users` 行（**首个用户自动 admin**，其余默认 annotator；`ZLSERVER_BOOTSTRAP_ADMIN/PASSWORD` 只建不改）。
3. 生成 32 字节随机 token（DB 只存 `sha256`），TTL 30 天。
4. 后续请求 `Authorization: Bearer <token>` → `get_auth` 依赖解析会话（60s 内存缓存命中校验）。
5. 会话失效/被吊销 → 401 `unauthorized`（`session_stale` 保留为客户端重新登录的语义码）。
6. 角色矩阵：`annotator` 领取/保存/提交自己的任务；`reviewer` 额外可 `force` 释放、复核、扫描、标签写；`admin` 额外可建项目、改角色、管账号。`ZLSERVER_PROJECT_ACCESS_MODE=strict` 时改看 `project_members` 的项目级角色（全局 admin 例外）。
7. 改角色/停用/改密都会立即吊销该账号的全部会话（`AuthService.update_user/set_password`，REST、CLI、`/admin` 共用同一条路径）。

## 8. 推理服务（独立进程）

```
桌面端 ─▶ POST /api/v2/projects/{p}/predict ─▶ InferenceClient ─HTTP(内部 token)─▶ inference worker
                                                  │                                 ├─ 模型（sam_ort/，原样搬）
                                                  └─ 取图（存储树，按需）            ├─ embedding 缓存（key=image sha256）
                                                                                    └─ 单/多 GPU 队列
```

- job（API → worker，`POST /infer`，Bearer 内部 token）：`{job_id, anno_id, image_sha256, image_b64, model, prompts{points,labels,rects,texts}, threshold, mode, return_type, crop_box}`；同步等待（HTTP，超时 `ZLSERVER_INFERENCE_TIMEOUT`），worker 内部串行或按 GPU 并发。M3 可再加 `image_url` 拉取模式，避免大图重复上传。
- **embedding 缓存 key = 图像 sha256（+ crop）**：runner 暴露 `export_image_state()/import_image_state()`，把编码结果（SAM 的 `image_embeddings`、SAM2 的 `image_embed/high_res_*`、SAM3 的 `img/pcs/pvs feats`）整体快照；命中时**直接恢复**而非重算（`EmbeddingCache`，LRU，`ZLSERVER_EMBEDDING_CACHE_SIZE`）。同一张图第二次点击零编码，且每个 job 只可能用自己那帧的编码。
- worker 暴露 `GET /health`（模型/设备/是否已加载/缓存与队列深度，无需鉴权）与 `GET /metrics`（jobs/errors/命中率/p50-p95/max/uptime，需内部 token）。
- 取图两种方式：默认 `ZLSERVER_INFERENCE_INLINE_IMAGES=true` 随 job 内联；置 false 则 API 把帧写进内容寻址缓存并给 `image_url`，worker 在**缓存未命中时**才拉（`GET /api/v2/internal/images/{sha}`，内部 token 鉴权）。
- `mode` 校验前置（不再 500）：点提示只接受 1(SAM)/2(CV)，框提示接受 0/1/2/3，文本只接受 1；`mode=0` 是客户端"同时勾选 SAM+OpenCV"的历史值，按旧语义交给框路径。
- 模型参数沿用 v1：`ZLSERVER_MODEL_NAME/DIR/BACKEND`、SAM3 conf/iou、轮廓后处理参数（与桌面端逐像素对齐的预处理逻辑保持不变）。
- 降级：worker 不可达 → `503 inference_unavailable`，客户端提示"推理不可用，可继续手动标注"（不再 500）。

## 9. 序列分组：由服务端解析

沿用桌面端现有规则（`species/dish/D{n}.png`），在扫描时写入 `tasks.group_name/day`：

- 末段匹配 `D(\d+)\.(png|jpg|jpeg)` → `day=n`；路径 ≥3 段时 `group_name = 倒数第3段/倒数第2段`，2 段时 `group_name = 倒数第2段`（与客户端 `_assign_remote_group` 完全一致）。
- 否则回退 `(.+?)[_\- ]*(\d+)\.(ext)` → `group_name=前缀, day=序号`；都不匹配则 `group_name="", day=0`。
- v2 客户端直接用服务端字段，删除客户端的文件名猜测；v1 兼容层保持旧返回（不带 group/day）。

## 10. v1 的删除与不可回退点

v1 代码已整体删除，`onnx` 分支（提交 `fc04ef4`）是唯一留档。搬迁/删除边界：

**搬走（继续服务 v2）**
- `app/sam_ort/`、`app/worker.py`、`app/ztypes.py`、`app/bbox_overlaps.py`、`app/debug_save.py` → `inference/`（模型参数与 logger 各自成模块：`inference/config.py`、`inference/logging.py`）
- 数值回归测试（preprocess/postprocess/tokenize/worker/backends/models_slow/cuda）→ `tests/inference/`

**删除（v1 专属）**
- `app/app.py`（v1 全部路由与模块级状态）、`app/db.py`、`app/config.py`、`app/logger.py`、`app/project_scan.py`、`app/schemas.py`
- v1 的接口/DB 语义测试（test_api / test_auth / test_labels / test_save_zlabel / test_missing_tasks / test_anno_id / test_project_scan）——覆盖在 M2 用 v2 服务层测试重建；`anno_id` 契约已在 `tests/v2/test_models.py` 固化
- 未使用依赖 `fastapi-users`、`typer`；`.env.onnx`（v1 变量名）→ `.env.example`（`ZLSERVER_*`）

**不可回退点**：`/api/v1` 与 `zlabel_server.db` 一起失效。回滚只有两条路——部署上一个 v2 版本（推荐，数据面不变），或整体退回 `onnx` 分支（必须同时退回旧客户端，且新旧库数据不互通）。

## 11. 可观测性与运维

- 结构化日志 + `X-Request-ID`；`audit_log` 记录领取/提交/复核/强制覆盖/标签变更。
- `GET /api/v2/health` 汇总 DB / 存储根 / 推理进程；`/metrics` 暴露推理延迟与缓存命中率。
- `docker-compose.yml` 增加 `zlabel_inference` 服务（同镜像不同 entrypoint，共享 `assets/onnx`、保留 GPU 预留），`zlabel_server` 走内部网络访问。

## 12. 迁移与灰度

| 步 | 动作 | 出口标准 |
|---|---|---|
| M0 | v1 冻结基线（已做：`fc04ef4`） | `pytest -m "not slow"` 除真 GPU 用例全绿 |
| M1 | v2 骨架（config/db/models/security/health）+ alembic 初始迁移 | ✅ 已完成：`/api/v2/health` 可用，`upgrade head`/`downgrade base`/`alembic check` 全绿 |
| M1b | 资产搬迁（`inference/`、`v2/vendor/`）+ v1 彻底删除 + 基建/文档更新 | ✅ 已完成：`app/` 不存在，推理回归测试全绿 |
| M2 | 服务层 + `/api/v2` 端点（auth/projects/tasks/annotations/images/labels/progress/predict）+ hermetic 测试 | ✅ 已完成：109 个 v2 用例（role/lease/conflict/版本/预测） |
| M3 | 推理进程 + InferenceClient（embedding 缓存、health、metrics） | ✅ 已完成：同图重复请求零编码（快照恢复），模型在独立进程 |
| M4 | 桌面端切 v2（见 `client-migration-v2.md`） | 进行中：协议层 C1–C3+C7 已完成（分支 `zlabel-v2-client`，含跨仓库契约测试）；领取/复核 UI 等留待第二批 |
| M5 | 验收：DoD §14 全项通过 | 交付 |

回滚：v1 已删除，v2 与 v1 不互通，因此切换需要一个明确的停机窗口。窗口内回滚 = 部署上一个 v2 版本（数据面不变，v2 从 M1 起就一直用新库）；只有放弃 v2 时才退回 `onnx` 分支 + 旧客户端。

## 13. 测试策略

- **单元**：services（领取/租约/状态机/角色/冲突）、分组解析、schema 校验。
- **API（hermetic）**：临时目录里的真实本地存储 + 假推理；覆盖 401/403/404/409/租约/校验。
- **契约 golden**：把桌面端真实请求 payload（form / JSON）固化为夹具，客户端升级时作为回归基线（v1 服务端已删除，改为夹具对比）。
- **数值回归**：`tests/inference/`（preprocess/postprocess/tokenize/worker/backends/models_slow），搬迁后原样通过。

## 14. 验收标准（DoD）

1. 两个标注员同时打开同一帧：第二个立刻拿到 409 + 持有者信息，UI 可等待/跳转；租约到期后可自动领取。
2. 一次推理调用只依赖它自己请求的那张图（同一会话内交替预测两张图，结果与单图预测一致）。
3. `reviewer` 可复核退回，`annotator` 不能；越权返回 403 且写入审计。
4. 版本冲突：客户端 A 保存后，B 用旧 `base_version` 保存 → 409 带服务端版本与作者；`force` 仅 reviewer+。
5. 进度 `{total,draft,submitted,approved,rejected}` 与任务状态一致，`by_user` 与审计一致。
6. 推理进程挂掉/重启：存取/复核照常，仅 predict 返回 503 且客户端有可读提示。
7. 桌面端（v2 版）与服务器同步升级后，登录/取任务/取图/读写标注/复核/推理全链路可用；旧版客户端连 v2 服务器会被明确拒绝（版本不匹配）。
8. 除真 GPU 用例外的测试全绿；alembic 可从空库升到最新。

## 15. 已知风险

- **账号丢失/遗忘密码**：服务端不保存明文，忘记密码只能由管理员在 `/admin/accounts` 重置（或 `uv run python -m v2.cli user passwd`）；没有自助找回。
- **SQLite 写并发**：领取/提交是短事务，<50 人够用；上多副本需换 Postgres（仓库层已隔离，切换成本可控）。
- **SAM3 显存**：vision arena ~7GB 不回收，worker 每帧建/销 session 的既有策略必须逐行保留。
- **无灰度能力**：`/api/v1` 已删除，服务端与桌面端必须同步发布；上线前要有停机窗口与“上一版 v2”的部署包。
- **历史元数据不保留**：`.zlabel` 标注文件格式不变（历史标注可继续读取），但旧库里的领取/进度/标签元数据按决策丢弃。

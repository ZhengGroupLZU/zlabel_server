# ZLabel Server v2 — 架构设计

> 状态：设计已定，实现中。v1（分支 `onnx`，提交 `fc04ef4`）为参考实现，保持可运行。

## 1. 目标与非目标

**目标**：把标注服务端从"单用户脚本式后端"升级为**多用户生产系统**：任务领取与租约、提交/复核状态机、角色权限与审计、按人统计、服务端标签管理、推理与 API 解耦（不再有"全局当前帧"），并且 HTTP 契约有版本、可灰度、可回滚。

**非目标（v2 不做）**：不做 Web 前端；不改桌面端的本地（离线）模式；不替换 OpenList 作为文件存储；不引入 Postgres/Redis 等外部中间件（保持单机可部署，但边界留好）。

## 2. 已确认的四项决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | **OpenList 作身份源 + 服务端签发 session**：登录仍用 OpenList 账号，服务端查 OpenList 校验后签发自己的 token，`users` 表按 OpenList 用户 id 存角色/统计，并把该用户的 OpenList token 存在会话里用于文件访问 | 保留 OpenList 的按用户 ACL；同时获得本地角色/领取/审计能力；桌面端登录字段不变 |
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
    config.py            # pydantic-settings，ZLV2_* 前缀
    logging.py           # 结构化日志 + request id
    errors.py            # 统一错误模型 {code, message, detail}
    security.py          # session 签发/校验、角色依赖（require_user / require_role）
  db/
    base.py              # engine/session 工厂
    models.py            # SQLAlchemy 模型（见 §4）
    repositories/        # 每个聚合一个仓库（projects/tasks/annotations/users/labels/audit）
    migrations/          # alembic
  vendor/openlist_api/   # vendored 第三方 OpenList SDK（原 app/openlist_api）
  adapters/
    openlist.py          # 唯一出口：包装 vendor/openlist_api + 会话 token 注入
    inference.py         # InferenceClient（HTTP 调推理进程）
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

**全新数据库**（`ZLV2_DATABASE_URL`，默认 `data/zlabel_server_v2.db`）：
v2 **不迁移、不导入 v1 数据**（旧项目不保留；任务表本来就能由 OpenList 重扫重建）。
v1 的 `zlabel_server.db` 留给旧服务端（M4 双跑对比需要两者并存）。
schema 由 **alembic** 管理（`uv run alembic upgrade head`），首次即建全量 v2 表：

```
users            id, oplist_user_id(uniq), name(uniq, lower), email, role[annotator|reviewer|admin],
                 active, created_at, last_login_at, finished_count
sessions         id, token_hash(uniq), user_id, oplist_token, created_at, expires_at, revoked_at, client_info
projects         id, name(uniq, = OpenList 目录名), display_name, description, active, marker_ok,
                 created_at, updated_at
labels           id, project_id, name, color, sort, archived, created_at   (uniq(project_id,name))
tasks            id, project_id, anno_id(uniq), path(OpenList 绝对路径), rel_path, group_name, day,
                 state[draft|submitted|approved|rejected], missing,
                 claimed_by, claimed_at, lease_expires_at,
                 submitted_at, reviewed_by, reviewed_at, review_note, updated_at
annotations      id, task_id(uniq), anno_id, version(int), author_id, path, labels_json, hash, created_at, updated_at
annotation_versions  id, task_id, version, author_id, path, labels_json, created_at, note   (历史)
link_task_label  (task_id, label_id)      # 保留，v1 兼容读
link_task_user   (task_id, user_id)       # 保留
audit_log        id, ts, user_id, action, target_type, target_id, detail_json
```

要点：

- **`tasks.anno_id` 公式不变**：`md5("<project>/<project-relative posix path>")`。历史标注与桌面端本地镜像必须继续对得上。
- **标注文件仍在 OpenList**（`{proj_dir}/{project}/zlabel/{anno_id}.zlabel`），DB 只存元数据/版本/哈希 → 与桌面端镜像互通不变。
- `tasks.rel_path` 是**项目内相对路径**（`images/a/D1.png`），`tasks.path` 是 OpenList 绝对路径；`get_image` 一律走 `project + rel_path`，修掉 v1 的 `get_image` 忽略 `project` 的口径问题。
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
| `GET /api/v2/health` | — | `{db, openlist, inference, version, capabilities[]}`（能力探测用） |
| `GET /api/v2/projects` | session | 项目列表（含我的进度） |
| `POST /api/v2/projects` | admin/reviewer | 建项目（OpenList 目录 + 标记文件） |
| `PATCH /api/v2/projects/{p}` | admin/reviewer | 描述/显示名/启停 |
| `POST /api/v2/projects/{p}/scan` | reviewer/admin | 重扫 OpenList 同步任务（替代 `refresh_tasks`） |
| `GET/POST/PATCH/DELETE /api/v2/projects/{p}/labels[/{id}]` | 读 session / 写 reviewer+ | 标签 CRUD（名字、颜色、排序、归档） |
| `GET /api/v2/projects/{p}/tasks` | session | `state,claim,group,mine,limit,cursor,order=id\|sequence` |
| `POST /api/v2/tasks/{anno_id}/claim\|release\|heartbeat` | session | 领取/释放/续租 |
| `POST /api/v2/tasks/{anno_id}/submit` | 持有者 | 提交复核 |
| `POST /api/v2/tasks/{anno_id}/review` | reviewer/admin | `{decision: approve\|reject, note}` |
| `GET /api/v2/projects/{p}/groups` | session | 序列分组 + 帧列表（时间轴/拷贝上一帧用，服务端给全量，不受分页截断） |
| `GET /api/v2/projects/{p}/my-stats` | session | 我名下各状态的任务数 |
| `POST /api/v2/tasks/{anno_id}/reopen` | reviewer/admin | 已通过/已提交退回草稿（复核者改主意） |
| `GET /api/v2/auth/users` · `PUT /api/v2/auth/users/{id}/role` | admin | 用户列表 / 改角色（改角色会撤销该用户会话） |
| `GET /api/v2/projects/{p}/images/{rel_path:path}` | session | 取图（用**会话用户自己的** OpenList token），支持 `ETag`/`If-None-Match`；**无副作用** |
| `GET /api/v2/images/{sha256}` | session | 取回客户端上传过（内容寻址）的帧 |
| `PUT /api/v2/projects/{p}/images/{rel_path:path}` | session | 本地上传（本地数据集 + 远端推理） |
| `GET /api/v2/projects/{p}/annotations/{anno_id}` | session | 200 + `ETag: v{n}` / 404 = 未标注 |
| `PUT /api/v2/projects/{p}/annotations/{anno_id}` | 持有者 | body 含 `base_version`；200 `{version}` / 409 冲突（`server_version, updated_by, updated_at`）/ `force=true` 仅 reviewer+ |
| `GET /api/v2/projects/{p}/annotations/{anno_id}/versions[/{n}]` | session | 版本历史 / 某一版内容（历史文件存 `zlabel/_history/<anno_id>/v<n>.zlabel`） |
| `POST /api/v2/projects/{p}/predict` | session | 交互/文本推理，**无状态**：multipart `data`(JSON) + 可选 `image`；`data` 里可带 `rel_path`（服务端去 OpenList 取）或 `image_sha256`（取上传缓存） |
| `GET /api/v2/projects/{p}/progress` | session | `{total,draft,submitted,approved,rejected}`，`by_user=true` 时带按人明细 |

错误码约定：`404 not_found`（未标注/未找到，客户端可安全新建）、`409 conflict`（版本冲突或任务被他人领取，`detail.claimed_by` 区分）、`502 upstream_error`（OpenList/推理进程失败）、`503 inference_unavailable`。

## 7. 鉴权与会话

1. 客户端 `POST /api/v2/auth/login`（字段与现在一致：用户名/密码）。
2. 服务端调 OpenList `auth.login` 校验 → 取用户 id/name → `upsert users`（**首个用户自动 admin**，其余默认 annotator；可由 admin 改角色）。
3. 生成 32 字节随机 token（DB 只存 `sha256`），TTL 30 天；把该用户的 OpenList token 存入 `sessions.oplist_token`。
4. 后续请求 `Authorization: Bearer <token>` → `require_user` 依赖解析会话（60s 内存缓存命中校验），得到 `User` + 可用于 FS 的 OpenList token。
5. OpenList 侧 token 失效 → FS 调用返回 401 `session_stale`，客户端重新登录（服务端**不保存密码**）。
6. 角色矩阵：`annotator` 领取/保存/提交自己的任务；`reviewer` 额外可 `force` 释放、复核、扫描、标签写；`admin` 额外可建项目、改角色。

## 8. 推理服务（独立进程）

```
桌面端 ─▶ POST /api/v2/projects/{p}/predict ─▶ InferenceClient ─HTTP(内部 token)─▶ inference worker
                                                  │                                 ├─ 模型（sam_ort/，原样搬）
                                                  └─ 取图（OpenList，按需）           ├─ embedding 缓存（key=image sha256）
                                                                                    └─ 单/多 GPU 队列
```

- job（API → worker，`POST /infer`，Bearer 内部 token）：`{job_id, anno_id, image_sha256, image_b64, model, prompts{points,labels,rects,texts}, threshold, mode, return_type, crop_box}`；同步等待（HTTP，超时 `ZLV2_INFERENCE_TIMEOUT`），worker 内部串行或按 GPU 并发。M3 可再加 `image_url` 拉取模式，避免大图重复上传。
- **embedding 缓存 key = 图像 sha256（+ crop）**：runner 暴露 `export_image_state()/import_image_state()`，把编码结果（SAM 的 `image_embeddings`、SAM2 的 `image_embed/high_res_*`、SAM3 的 `img/pcs/pvs feats`）整体快照；命中时**直接恢复**而非重算（`EmbeddingCache`，LRU，`ZLV2_EMBEDDING_CACHE_SIZE`）。同一张图第二次点击零编码，且每个 job 只可能用自己那帧的编码。
- worker 暴露 `GET /health`（模型/设备/是否已加载/缓存与队列深度，无需鉴权）与 `GET /metrics`（jobs/errors/命中率/p50-p95/max/uptime，需内部 token）。
- 取图两种方式：默认 `ZLV2_INFERENCE_INLINE_IMAGES=true` 随 job 内联；置 false 则 API 把帧写进内容寻址缓存并给 `image_url`，worker 在**缓存未命中时**才拉（`GET /api/v2/internal/images/{sha}`，内部 token 鉴权）。
- `mode` 校验前置（不再 500）：点提示只接受 1(SAM)/2(CV)，框提示接受 0/1/2/3，文本只接受 1；`mode=0` 是客户端"同时勾选 SAM+OpenCV"的历史值，按旧语义交给框路径。
- 模型参数沿用 v1：`ZLV2_MODEL_NAME/DIR/BACKEND`、SAM3 conf/iou、轮廓后处理参数（与桌面端逐像素对齐的预处理逻辑保持不变）。
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
- `app/openlist_api/`（vendored SDK）→ `v2/vendor/openlist_api/`，由 `v2/adapters/openlist.py` 唯一引用
- 数值回归测试（preprocess/postprocess/tokenize/worker/backends/models_slow/cuda）→ `tests/inference/`

**删除（v1 专属）**
- `app/app.py`（v1 全部路由与模块级状态）、`app/db.py`、`app/config.py`、`app/logger.py`、`app/project_scan.py`、`app/schemas.py`
- v1 的接口/DB 语义测试（test_api / test_auth / test_labels / test_save_zlabel / test_missing_tasks / test_anno_id / test_project_scan）——覆盖在 M2 用 v2 服务层测试重建；`anno_id` 契约已在 `tests/v2/test_models.py` 固化
- 未使用依赖 `fastapi-users`、`typer`；`.env.onnx`（v1 变量名）→ `.env.example`（`ZLV2_*`）

**不可回退点**：`/api/v1` 与 `zlabel_server.db` 一起失效。回滚只有两条路——部署上一个 v2 版本（推荐，数据面不变），或整体退回 `onnx` 分支（必须同时退回旧客户端，且新旧库数据不互通）。

## 11. 可观测性与运维

- 结构化日志 + `X-Request-ID`；`audit_log` 记录领取/提交/复核/强制覆盖/标签变更。
- `GET /api/v2/health` 汇总 DB / OpenList / 推理进程；`/metrics` 暴露推理延迟与缓存命中率。
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
- **API（hermetic）**：假 OpenList（沿用 `v2/vendor/openlist_api` 的形状）+ 假推理；覆盖 401/403/404/409/租约/校验。
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

- **OpenList token 生命周期**：会话保存用户 token，OpenList 侧过期/改密会导致 FS 401，需客户端重新登录（已定义 `session_stale`）。
- **SQLite 写并发**：领取/提交是短事务，<50 人够用；上多副本需换 Postgres（仓库层已隔离，切换成本可控）。
- **SAM3 显存**：vision arena ~7GB 不回收，worker 每帧建/销 session 的既有策略必须逐行保留。
- **无灰度能力**：`/api/v1` 已删除，服务端与桌面端必须同步发布；上线前要有停机窗口与“上一版 v2”的部署包。
- **历史元数据不保留**：`.zlabel` 标注文件格式不变（历史标注可继续读取），但旧库里的领取/进度/标签元数据按决策丢弃。

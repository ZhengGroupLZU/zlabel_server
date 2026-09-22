# 去 OpenList 化：自建存储与账号（P0–P6）

决策（2026-09）：**本地/共享目录树存储 + 自建密码账号 + REST/CLI 先行 + 标注统一到 `.zlabel/annos/`**。
目标：服务端不再依赖外部文件服务即可完成全部功能（存储、账号、权限、扫描）。

## 已完成

| 阶段 | 内容 | 证据 |
|---|---|---|
| P0 存储抽象 | `v2/adapters/storage.py`：`StorageBackend` 协议（路径助手 + 8 个 IO 方法）+ `build_storage()`；OpenList 与本地两种实现由 `ZLSERVER_STORAGE_BACKEND` 选择 | `tests/v2/test_storage_contract.py`（同一套契约跑两个后端） |
| P1 本地存储 | `v2/adapters/local_disk.py`：原子写（temp+rename）、ETag=size+mtime、路径穿越/符号链接防护、`list_dirs/glob_images/ensure_dir/usage/delete` | 同上 + `tests/v2/test_local_backend_api.py` |
| P2 项目发现 | 不再依赖标记文件：本地后端下**每个顶层目录都是项目**（服务端拥有整棵树）；OpenList 仍按标记文件（`uses_marker_discovery`） | 同上（含"目录消失即停用"用例） |
| P3 自建账号 | `v2/adapters/identity.py`：`IdentityProvider` 协议 + `LocalIdentity`（**scrypt**，标准库，无新依赖）+ `OpenListIdentity`（过渡期）；`users.password_hash`（alembic `8394f2f1aff0`）；启动引导 `ZLSERVER_BOOTSTRAP_ADMIN/PASSWORD`（只建不改） | `tests/v2/test_identity.py` |
| P4 项目成员 | `project_members` 表（迁移 `045be3078645`）+ `ProjectService.role_for/require_access/require_project_reviewer` + 成员 CRUD 端点；`ZLSERVER_PROJECT_ACCESS_MODE=open\|strict`（默认 open 不改变现状）；项目列表/任务/标注/标签/复核全部按成员过滤，复核与标签写用**项目级**角色 | `tests/v2/test_members.py` |
| P5 REST + CLI | `/api/v2/admin/users`（建号/改角色/启停/改密）、`/api/v2/admin/storage`、`/api/v2/admin/files`（list/upload/download/delete/mkdir/move，仅本地后端）；CLI 增 `project ls\|members\|add-member` | `tests/v2/test_admin.py` + CLI 实测 |

零 OpenList 端到端（本地账号 + 本地存储，含领取/保存/提交/复核/进度）：`tests/v2/test_local_backend_api.py::test_zero_openlist_end_to_end`。

## 待做

- **P5 Web 后台（首版已上线）**：`/admin` 用 `starlette-admin` 挂载（`v2/admin/`），cookie 会话复用 `AuthService`（仅 admin 角色，CSRF 由库内置）。
  页面：Dashboard（存储/项目/任务状态）、Users（角色/启停）、Projects（仅元数据）、Members（P4）、Labels、只读 Frames 与 Audit log。
  仍待补：**文件浏览器/上传**（本地后端）、项目创建与扫描按钮、密码重置页（这些目前走 CLI 或 `/api/v2/admin/*`）；
  因为页面按服务层能力拆分，后续想换成自研 Jinja2+HTMX 可以逐页替换。
- **P6 下线 OpenList**：删除 `v2/vendor/openlist_api/`（≈2000 行）与 `OpenListIdentity/OpenListAdapter`，`ZLSERVER_STORAGE_BACKEND`/`IDENTITY` 开关一并移除。

## 权限模型（P4）

- `ZLSERVER_PROJECT_ACCESS_MODE=open`（默认）：任何账号可使用任何项目，角色取全局角色 —— 与 P4 之前完全一致。
- `strict`：非管理员只能看到/操作 `project_members` 里的项目；**项目角色优先于全局角色**（全局 admin 例外，始终可见全部）。
- 复核、标签写、单项目扫描需要**该项目**的 reviewer；全局 reviewer 仍可扫描整棵树（`POST /projects/scan`）与建项目。
- 服务层的角色检查通过 `AuthContext.with_role()` 注入的项目角色生效，因此复核/强制覆盖等内部判断自动跟着项目走。

## 存储布局

```
<ZLSERVER_STORAGE_ROOT>/
  <project>/                                  # 任意目录结构（images/... 由使用者决定）
    .zlabel/
      project.json                            # 可选元数据（与桌面端本地数据集一致）
      annos/<anno_id>.zlabel                  # 标注
      annos/_history/<anno_id>/v<n>.zlabel    # 版本历史
    .zlabel-server-project-root               # 兼容 OpenList 部署的标记文件（本地后端不要求）
```
`anno_id = md5("<project>/<项目内相对路径>")` 不变；桌面端**按 anno_id 取文件**，所以布局变化对客户端完全透明。
目录名可通过 `ZLSERVER_ANNO_DIR` 覆盖；**留空时按后自动选择**：openlist 后端用历史上的 `zlabel`（现有部署不受影响），local 后端用 `.zlabel/annos`。搬完目录后显式设成 `.zlabel/annos`。

## 从 OpenList 迁移到本地存储（runbook）

1. 把 OpenList 的数据目录挂到服务器（或 rsync 过来），设为 `ZLSERVER_STORAGE_ROOT`。
2. **先保持** `ZLSERVER_STORAGE_BACKEND=openlist` 不动（`ZLSERVER_ANNO_DIR` 留空即用历史布局），确认服务端能读。
3. `uv run python -m v2.cli migrate-layout --root <STORAGE_ROOT> --dry-run` 看计划，确认后去掉 `--dry-run`。
4. `ZLSERVER_STORAGE_BACKEND=local`（此时自动用 `.zlabel/annos`），重启 → 扫描一次（`POST /projects/scan`）即可看到全部任务。
5. 账号：`ZLSERVER_IDENTITY=local` + `ZLSERVER_BOOTSTRAP_ADMIN/PASSWORD` 建首个管理员；之后用
   `uv run python -m v2.cli user add <名字> --role reviewer|annotator` 建人。
6. 回滚：反向运行 `migrate-layout --source .zlabel/annos --target zlabel`，并把两个开关切回。

## 必须配套的运维

- **备份**：数据现在是自家盘上的普通文件 → 文件系统快照/`rsync --hard-links` + SQLite `VACUUM INTO`（保证 DB 与文件同一时点）。
- **磁盘**：监控剩余空间；`ZLSERVER_MAX_UPLOAD_BYTES` 限制单帧上传。
- **权限**：进程用户必须对 `STORAGE_ROOT` 有写权限；符号链接与 `..` 已被拒绝。
- **安全**：登录已限流失败即拒（scrypt 校验），后续可加失败计数与审计告警。

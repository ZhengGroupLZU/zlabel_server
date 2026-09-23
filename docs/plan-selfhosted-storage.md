# 去 OpenList 化：自建存储与账号（P0–P6）✅

决策（2026-09）：**本地/共享目录树存储 + 自建密码账号 + REST/CLI 先行 + 标注统一到 `.zlabel/annos/`**。
目标：服务端不再依赖外部文件服务即可完成全部功能（存储、账号、权限、扫描）。

> **状态：P0–P6 全部完成。** `v2/vendor/openlist_api/`、`v2/adapters/openlist.py`、`OpenListIdentity`
> 与 `ZLSERVER_STORAGE_BACKEND` / `ZLSERVER_IDENTITY` / `ZLSERVER_OPLIST_*` 开关已删除，
> `users.oplist_user_id` → `identity_id`、`sessions.oplist_token` 已由迁移 `4f7c1d2ab9e3` 处理。
> 另一台服务器上仍跑着 OpenList 老版本的部署，按下方 runbook 迁移数据即可。

## 已完成

| 阶段 | 内容 | 证据 |
|---|---|---|
| P0 存储抽象 | `v2/adapters/storage.py`：`StorageBackend` 协议（路径助手 + IO 方法）+ `build_storage()` | `tests/v2/test_local_backend_api.py` |
| P1 本地存储 | `v2/adapters/local_disk.py`：原子写（temp+rename）、ETag=size+mtime、路径穿越/符号链接防护、`list_dirs/glob_images/ensure_dir/usage/delete` | 同上 |
| P2 项目发现 | 不再依赖标记文件：**每个顶层目录都是项目**（服务端拥有整棵树） | 同上（含"目录消失即停用"用例） |
| P3 自建账号 | `v2/adapters/identity.py`：`IdentityProvider` 协议 + `LocalIdentity`（**scrypt**，标准库，无新依赖）；`users.password_hash`（alembic `8394f2f1aff0`）；启动引导 `ZLSERVER_BOOTSTRAP_ADMIN/PASSWORD`（只建不改） | `tests/v2/test_identity.py` |
| P4 项目成员 | `project_members` 表（迁移 `045be3078645`）+ `ProjectService.role_for/require_access/require_project_reviewer` + 成员 CRUD 端点；`ZLSERVER_PROJECT_ACCESS_MODE=open\|strict`（默认 open 不改变现状）；项目列表/任务/标注/标签/复核全部按成员过滤，复核与标签写用**项目级**角色 | `tests/v2/test_members.py` |
| P5 REST + CLI | `/api/v2/admin/users`（建号/改角色/启停/改密）、`/api/v2/admin/storage`、`/api/v2/admin/files`（list/upload/download/delete/mkdir/move，仅本地后端）；CLI 增 `project ls\|members\|add-member` | `tests/v2/test_admin.py` + CLI 实测 |
| P5b Web 后台 | `/admin`（`starlette-admin` + 手写页面，cookie 会话复用 `AuthService`，仅 admin 角色）：**Dashboard**（存储/进度/项目表/重扫）、**Users**（筛选/建号/改角色/启停/改密）、**Projects**（列表筛选/新建；详情页：概览重命名与元数据、文件浏览上传预览删除、成员、标签、帧表带状态筛选）、**Files**（整树浏览）、只读 **Audit log** | `tests/v2/test_admin_ui.py` |
| P6 下线 OpenList | 删除 vendor SDK（≈2000 行）+ `OpenListAdapter`/`OpenListIdentity` + 全部开关；storage 协议不再透传 token；marker 发现与 health 的 openlist 探测移除；DB 列改名（迁移 `4f7c1d2ab9e3`） | 全量测试在本地后端上绿 |

自建端到端（本地账号 + 本地存储，含领取/保存/提交/复核/进度）：`tests/v2/test_local_backend_api.py::test_self_hosted_end_to_end`。

## 后台写操作的服务层约束

后台页面（手写 HTML，`v2/admin/views.py`）直接调服务层，与 API 同一套规则：

- **Users 页**（建号/改角色/启停/改密）调用 `AuthService.create_user/update_user/set_password`：
  写审计行，并在降权/停用时**立即吊销目标账号的全部会话**。
- **Projects 详情页**的成员、标签、项目元数据分别调 `ProjectService.add_member/remove_member`、
  `create_label/update_label/delete_label`、`update_project`（都带 `actor_id` 审计）。
- 文件操作走 `StorageBackend`（项目页内限制在项目目录下，`_clean_rel` 去掉 `..`）。
- 唯一的 `ModelView` 是只读 Audit log；`can_*` 钩子必须写成**同步**方法
  （starlette-admin 1.x 同步调用它们；`async def` 会让权限检查恒为真）。
- 手写页面通过 `index.html` 渲染，`widget_html` 必须包 `Markup`，否则会被 Jinja 转义。

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
```
`anno_id = sha256("<项目 key>/<项目内相对路径>")`（key = 数据集 `.zlabel/project.json` 的 `"id"`，服务端存在 `projects.key`）；桌面端**按 anno_id 取文件**，所以布局变化对客户端完全透明，改目录名/显示名也不影响标注。旧版 `md5("<项目名>/<路径>")` 文件用 `uv run python -m v2.cli migrate-anno-ids` 一次性改名。
目录名可通过 `ZLSERVER_ANNO_DIR` 覆盖，默认 `.zlabel/annos`（桌面端数据集布局）。老部署搬完目录后显式设成 `.zlabel/annos`。

## 从 OpenList 迁移到本地存储（runbook）

1. 把 OpenList 的数据目录挂到服务器（或 rsync 过来），设为 `ZLSERVER_STORAGE_ROOT`。
2. `uv run python -m v2.cli migrate-layout --root <STORAGE_ROOT> --dry-run` 看计划，确认后去掉 `--dry-run`：
   它把 `<project>/zlabel` 搬到 `<project>/.zlabel/annos`（历史版本跟着走）。
   `migrate-layout` 是纯文件系统操作，与后端实现无关。
3. 建账号：`ZLSERVER_BOOTSTRAP_ADMIN/PASSWORD` 建首个管理员；之后用
   `uv run python -m v2.cli user add <名字> --role reviewer|annotator` 建人。
4. 重启 → 扫一次（`POST /projects/scan`，或后台 Dashboard 页的 Rescan 按钮）；扫完会为每个项目在
   `.zlabel/project.json` 写入稳定的 `"id"`（anno_id 的命名空间）。
5. 旧标注改名：`uv run python -m v2.cli migrate-anno-ids --dry-run` 看计划，确认后去掉
   `--dry-run`（把 `md5("<项目名>/<路径>")` 的标注文件与历史版本改成 `sha256("<key>/<路径>")`，
   并同步 tasks/annotations/annotation_versions 行；桌面端本地数据集首次打开时也会自动改名）。
6. 校验：客户端重新扫描后，之前已标注的帧应能直接打开（anno_id 已换新名，内容不变）。
7. 标记文件 `.zlabel-server-project-root` 已不再需要，可以删掉（迁移不会动它）。
8. 回滚：反向运行 `migrate-layout --source .zlabel/annos --target zlabel`；如果要恢复 OpenList 后端，
   需要退回到 P6 之前的版本（本仓库不再提供该实现）。

## 必须配套的运维

- **备份**：数据现在是自家盘上的普通文件 → 文件系统快照/`rsync --hard-links` + SQLite `VACUUM INTO`（保证 DB 与文件同一时点）。
- **磁盘**：监控剩余空间；`ZLSERVER_MAX_UPLOAD_BYTES` 限制单帧上传。
- **权限**：进程用户必须对 `STORAGE_ROOT` 有写权限；符号链接与 `..` 已被拒绝。
- **安全**：登录已限流失败即拒（scrypt 校验），后续可加失败计数与审计告警。

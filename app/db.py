import hashlib

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.config import SETTINGS


class Base(DeclarativeBase):
    pass


link_task_label = Table(
    "link_task_label",
    Base.metadata,
    Column("task_id", ForeignKey("tasks.id"), unique=False),
    Column("label_id", ForeignKey("labels.id"), unique=False),
)

link_task_user = Table(
    "link_task_user",
    Base.metadata,
    Column("task_id", ForeignKey("tasks.id"), unique=False),
    Column("user_id", ForeignKey("users.id"), unique=False),
)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    tasks: Mapped[list["Task"]] = relationship("Task")

    def __repr__(self) -> str:
        return f"Project(id={self.id}, name={self.name}, tasks={self.tasks})"

    def add_task(self, task: "Task"):
        self.tasks.append(task)


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    anno_id: Mapped[str] = mapped_column(String, unique=True)
    filename: Mapped[str]
    labels: Mapped[list["Label"]] = relationship(
        secondary=link_task_label, back_populates="tasks"
    )
    finished: Mapped[bool]
    users: Mapped[list["User"]] = relationship(
        secondary=link_task_user, back_populates="tasks"
    )

    def __repr__(self) -> str:
        return f"Task(id={self.id}, anno_id={self.anno_id}, filename={self.filename}, labels={self.labels}, finished={self.finished}, users={self.users})"


class Label(Base):
    __tablename__ = "labels"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, unique=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String, unique=True)
    color: Mapped[str] = mapped_column(String, default="#000000")
    tasks: Mapped[list["Task"]] = relationship(
        secondary=link_task_label, back_populates="labels"
    )

    def __repr__(self) -> str:
        return f"Label(id={self.id}, name={self.name}, color={self.color})"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, unique=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String, unique=True)
    finished_count: Mapped[int] = mapped_column(Integer, default=0)
    tasks: Mapped[list["Task"]] = relationship(
        secondary=link_task_user, back_populates="users"
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, name={self.name}, finished_count={self.finished_count})"


engine = create_engine(f"sqlite+pysqlite:///{SETTINGS.database_url}")
session_maker = sessionmaker(engine, expire_on_commit=False)


def id_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def create_or_update_projects(projects: list[dict[str, str | list[str]]]):
    """
    [
        {
            "name": str,
            "files": list[str],
        }
    ]
    """
    batch: int = 1000
    with session_maker() as session:
        if session is None:
            return
        if not projects:
            return

        stmt_project = (
            insert(Project)
            .values([{"name": p["name"]} for p in projects])
            .on_conflict_do_nothing()
        )
        session.execute(stmt_project)

        for project in projects:
            project_files: list[str] = project["files"]  # type: ignore
            project_name: str = project["name"]  # type: ignore
            if not project_files:
                continue

            query = select(Project).where(Project.name == project_name)
            project_id = session.scalars(query).first()
            assert project_id is not None

            for i in range(0, len(project_files), batch):
                stmt_task = (
                    insert(Task)
                    .values(
                        [
                            {
                                "project_id": project_id.id,
                                "anno_id": id_md5(
                                    f"{SETTINGS.oplist_proj_dir}/{project_name}/{filename}"
                                ),
                                "filename": filename,
                                "finished": False,
                            }
                            for filename in project_files[i : i + batch]
                        ]
                    )
                    .on_conflict_do_nothing()
                )
                session.execute(stmt_task)

        session.commit()


def insert_data(
    task: dict | None = None,
    user: list[dict] | None = None,
    label: list[dict] | None = None,
):
    with session_maker() as session:
        if session is None:
            return
        if task and user:
            stmt_user = insert(User).values(user).on_conflict_do_nothing()
            session.execute(stmt_user)

            query = select(User).where(User.name.in_([t["name"] for t in user]))
            tmp = session.scalars(query)
            assert tmp is not None

            task_user = [{"task_id": task["id"], "user_id": t.id} for t in tmp]
            stmt_task_user = (
                insert(link_task_user).values(task_user).on_conflict_do_nothing()
            )
            session.execute(stmt_task_user)
        if task and label:
            stmt_label = insert(Label).values(label).on_conflict_do_nothing()
            session.execute(stmt_label)

            query = select(Label).where(Label.name.in_([t["name"] for t in label]))
            tmp = session.scalars(query)
            assert tmp is not None

            task_label = [{"task_id": task["id"], "label_id": t.id} for t in tmp]
            stmt_task_label = (
                insert(link_task_label).values(task_label).on_conflict_do_nothing()
            )
            session.execute(stmt_task_label)

        if task:
            stmt_task = insert(Task).values(task).on_conflict_do_nothing()
            session.execute(stmt_task)

        session.commit()


def finish_task(anno_id: str):
    with session_maker() as session:
        if session is None:
            return
        task = session.scalar(select(Task).where(Task.anno_id == anno_id))
        if task is not None and task.finished is False:
            task.finished = True
        session.commit()


def user_finished_count_plus(name: str):
    with session_maker() as session:
        if session is None:
            return
        user = session.scalar(select(User).where(User.name == name.lower()))
        if user is not None:
            user.finished_count += 1
        session.commit()


def insert_link_table(anno_id: str, label_name: str = "", user_name: str = ""):
    with session_maker() as session:
        if session is None:
            return
        task = session.scalar(select(Task).where(Task.anno_id == anno_id))
        if task is None:
            return

        label = session.scalar(select(Label).where(Label.name == label_name))
        if label is not None:
            stmt = insert(link_task_label).values(
                {"task_id": task.id, "label_id": label.id}
            )
            session.execute(stmt)
        user = session.scalar(select(User).where(User.name == user_name))
        if user is None:
            session.execute(insert(User).values(name=user_name))
            user = session.scalar(select(User).where(User.name == user_name))
        assert user is not None

        stmt = insert(link_task_user).values({"task_id": task.id, "user_id": user.id})
        session.execute(stmt)
        if task.finished is False:
            user.finished_count += 1
            task.finished = True
        session.commit()


def get_tasks(
    project: int = -1,
    num: int = 50,
    finished: int = 1,
    random: bool = True,
) -> list[Task]:
    """
    @param finished
        -1: all
        0: unfinished
        1: finished
    """
    with session_maker() as session:
        if session is None:
            return []
        query = select(Task)
        if project > -1:
            query = query.where(Task.project_id == project)
        if finished == -1:
            ...
        elif finished == 0:
            query = query.where(Task.finished == False)
        elif finished == 1:
            query = query.where(Task.finished == True)
        else:
            raise ValueError("finished must be -1, 0 or 1")
        if random:
            query = query.order_by(func.random()).limit(num)
        else:
            query = query.order_by(Task.id).limit(num)
        tasks = session.scalars(query).all()
        _ = [t.labels for t in tasks]
    return list(tasks)


def get_projects() -> list[Project]:
    with session_maker() as session:
        if session is None:
            return []
        query = select(Project).order_by(Project.id)
        projects = session.scalars(query).all()
        return list(projects)


def how_many_finished() -> int:
    with session_maker() as session:
        if session is None:
            return 0
        query = select(func.count()).select_from(Task).where(Task.finished == True)
        result = session.scalar(query)
        return result or 0


def get_task_by_anno_id(anno_id: str) -> Task | None:
    with session_maker() as session:
        if session is None:
            return None
        query = select(Task).where(Task.anno_id == anno_id)
        task = session.scalar(query)
        return task


Base.metadata.create_all(engine)


if __name__ == "__main__":
    create_or_update_projects(
        [
            {"name": "project1", "files": ["0.png", "1.png", "2.jpg"]},
            {"name": "project2", "files": ["1.png", "2.jpeg"]},
        ]
    )

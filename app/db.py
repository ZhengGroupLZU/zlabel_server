from typing import List

from fastapi import Depends
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy import (
    Integer,
    String,
    Column,
    create_engine,
    Table,
    ForeignKey,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert

DATABASE_URL = "sqlite+pysqlite:///./zlabel_server.db"


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


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anno_id: Mapped[str] = mapped_column(String)
    filename: Mapped[str]
    labels: Mapped[List["Label"]] = relationship(
        secondary=link_task_label, back_populates="tasks"
    )
    finished: Mapped[bool]
    users: Mapped[List["User"]] = relationship(
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
    tasks: Mapped[List["Task"]] = relationship(
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
    tasks: Mapped[List["Task"]] = relationship(
        secondary=link_task_user, back_populates="users"
    )

    def __repr__(self) -> str:
        return f"User(id={self.id}, name={self.name}, finished_count={self.finished_count})"


engine = create_engine(DATABASE_URL)
session_maker = sessionmaker(engine, expire_on_commit=False)


def insert_data(task, user, label):
    with session_maker() as session:
        if session is None:
            return
        if task is not None and label is not None:
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

        if task is not None and user is not None:
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

        if task is not None:
            stmt_task = insert(Task).values(task).on_conflict_do_nothing()
            session.execute(stmt_task)

        session.commit()


def update_task(anno_id):
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


def get_tasks(num: int = 50, finished: int = 1) -> List[Task]:
    with session_maker() as session:
        if session is None:
            return []
        query = select(Task).limit(num)
        if finished == -1:
            stmt = query
        elif finished == 0:
            stmt = query.where(Task.finished == False)
        elif finished == 1:
            stmt = query.where(Task.finished == True)
        else:
            raise ValueError("finished must be -1, 0 or 1")
        tasks = session.scalars(stmt).all()
        _ = [t.labels for t in tasks]
    return list(tasks)


Base.metadata.create_all(engine)

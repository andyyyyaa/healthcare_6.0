from functools import cache
import logging
import sqlite3
from typing import Any

def create_fracture(db_name: str) -> None:
    """Create database"""
    print("CREATE")
    with sqlite3.connect(f"{db_name}.sqlite3") as connection:
        cur = connection.cursor()
        cur.execute("DROP TABLE IF EXISTS fracture;")
        cur.execute(
            "CREATE table fracture ("
            + "name,"
            + "age,"
            +"sex,"
            + "fracture"
            + ");"
        )

def create_heart(db_name: str) -> None:
    """Create database"""
    print("CREATE")
    with sqlite3.connect(f"{db_name}.sqlite3") as connection:
        cur = connection.cursor()
        cur.execute("DROP TABLE IF EXISTS heart;")
        cur.execute(
            "CREATE table heart ("
            + "name,"
            + "age,"
            +"sex,"
            + "target"
            + ");"
        )

def create_user(db_name: str) -> None:
    """Create database"""
    print("CREATE")
    with sqlite3.connect(f"{db_name}.sqlite3") as connection:
        cur = connection.cursor()
        cur.execute("DROP TABLE IF EXISTS user;")
        cur.execute(
            "CREATE table user ("
            + "name,"
            + "password"
            + ");"
        )

create_fracture("user_db")
create_heart("user_db")
create_user("user_db")
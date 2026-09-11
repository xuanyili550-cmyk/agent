import json
import os
from student import Student
class StudentManager:
    def __init__(self):
        self.students=[]
        self.filename="student.json"
        self.load()

    def add(self):
        sid = input("学号：")
        name = input("姓名：")
        age = int(input("年龄："))
        score = float(input("成绩："))
        student = Student(sid, name, age, score)
        self.students.append(student)
        self.save()
        print("succes")

    def show(self):
        if len(self.students)==0:
            print("无数据")
            return
        for s in self.students:
            print(s)

    def save(self):
        data=[]
        for s in self.students:
            data.append(s.to_dict())
            with open(self.filename, "w", encoding="utf8") as f:
                json.dump(data,f, ensure_ascii=False, indent=4)

    def load(self):

        if not os.path.exists(self.filename):
            return

        with open(self.filename, "r", encoding="utf8") as f:
            data = json.load(f)

        for item in data:
            student = Student(
                item["student_id"],
                item["name"],
                item["age"],
                item["score"]
            )

            self.students.append(student)

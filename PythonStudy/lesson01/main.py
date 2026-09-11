from manager import StudentManager

manager = StudentManager()
while True:

    print("========学生管理系统========")
    print("1 添加学生")
    print("2 查看学生")
    print("0 退出")

    choice = input("请选择：")

    if choice == "1":
        manager.add()

    elif choice == "2":
        manager.show()

    elif choice == "0":
        break

    else:
        print("输入错误")



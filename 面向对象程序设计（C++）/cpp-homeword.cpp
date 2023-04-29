//所有的员工、经理等具有唯一UID

#include <iostream>
#include <cstring>
#include <fstream>
#include <windows.h>

#define sales "E://cpp//cpp//salesman.txt"
#define mana "E://cpp//cpp//manager.txt"
#define smana "E://cpp//cpp//salemanager.txt"
#define text1 "E://cpp//cpp//1.txt"
#define text2 "E://cpp//cpp//2.txt"

using namespace std;

class staff
{
protected:
    string age;
    string uid;
    string name;

public:
    staff()
    {
    }
    void show()
    {
    }
    string showifo(int i)
    {
    }
};

class salesman : virtual public staff
{
protected:
    string salesnum;

public:
    void input()
    {
        cout << "salesman:" << endl;
        cout << "请输入编号：" << endl;
        cin >> uid;
        cout << "请输入姓名：" << endl;
        cin >> name;
        cout << "请输入年龄：" << endl;
        cin >> age;
        cout << "请输入销售额：" << endl;
        cin >> salesnum;
    }
    void show()
    {
        cout << "编号：" << uid << endl;
        cout << "姓名：" << name << endl;
        cout << "年龄：" << age << endl;
        cout << "销售额：" << salesnum << endl;
    }
    string showifo(int i)
    {
        switch (i)
        {
        case 1:
            return uid;
        case 2:
            return name;
        case 3:
            return age;
        case 4:
            return salesnum;
        default:
            cout << "这是一次错误的输入" << endl;
            break;
        }
    }
};

class manager : virtual public staff
{
protected:
    string dep;

public:
    void input()
    {
        cout << "manager: " << endl;
        cout << "请输入编号：" << endl;
        cin >> uid;
        cout << "请输入姓名：" << endl;
        cin >> name;
        cout << "请输入年龄：" << endl;
        cin >> age;
        cout << "请输入部门：" << endl;
        cin >> dep;
    }
    void show()
    {
        cout << "编号：" << uid << endl;
        cout << "姓名：" << name << endl;
        cout << "年龄：" << age << endl;
        cout << "部门：" << dep << endl;
    }
    string showifo(int i)
    {
        switch (i)
        {
        case 1:
            return uid;
        case 2:
            return name;
        case 3:
            return age;
        case 4:
            return dep;
        default:
            cout << "这是一次错误的输入" << endl;
            break;
        }
    }
};
class salemanager : public manager, public salesman
{
protected:
    string title;

public:
    void input()
    {
        cout << "salesmanager" << endl;
        cout << "请输入编号：" << endl;
        cin >> uid;
        cout << "请输入姓名：" << endl;
        cin >> name;
        cout << "请输入年龄：" << endl;
        cin >> age;
        cout << "请输入销售额：" << endl;
        cin >> salesnum;
        cout << "请输入部门：" << endl;
        cin >> dep;
        cout << "请输入职称：" << endl;
        cin >> title;
    }
    void show()
    {
        cout << "编号：" << uid << endl;
        cout << "姓名：" << name << endl;
        cout << "年龄：" << age << endl;
        cout << "销售额：" << salesnum << endl;
        cout << "部门：" << dep << endl;
        cout << "职称：" << title << endl;
    }
    string showifo(int i)
    {
        switch (i)
        {
        case 1:
            return uid;
        case 2:
            return name;
        case 3:
            return age;
        case 4:
            return dep;
        case 5:
            return salesnum;
        case 6:
            return title;
        default:
            cout << "这是一次错误的输入" << endl;
            break;
        }
    }
};

void add()
{
    fstream file1, file2, file3;
    cout << "1：增加销售员：" << endl;
    cout << "2：增加经理：" << endl;
    cout << "3：增加销售经理：" << endl;
    cout << "4：返回" << endl;
    int i;
    string s;
    cin >> i;
    switch (i)
    {
    case 1:
    {
        file1.open(sales, ios::out | ios::app);
        if (!file1)
        {
            cout << "打开错误或者文件不存在";
            break;
        }
        salesman a;
        a.input();
        {
            for (int j = 1; j < 5; j++)
            {
                s = a.showifo(j);
                file1 << s;
                if (j < 4)
                {
                    file1 << " ";
                }
                if (j == 4)
                {
                    file1 << endl;
                }
            }
        }
        file1.close();
        break;
    }
    case 2:
    {
        file2.open(mana, ios::out | ios::app);
        if (!file2)
        {
            cout << "打开错误或者文件不存在";
            break;
        }
        manager b;
        b.input();
        {
            for (int j = 1; j < 5; j++)
            {
                s = b.showifo(j);
                file2 << s;
                if (j < 4)
                {
                    file2 << " ";
                }
                if (j == 4)
                {
                    file2 << endl;
                }
            }
        }
        file2.close();
        break;
    }
    case 3:
    {
        file3.open(smana, ios::out | ios::app);
        if (!file3)
        {
            cout << "打开错误或者文件不存在";
            break;
        }
        salemanager c;
        c.input();
        {
            for (int j = 1; j < 7; j++)
            {
                s = c.showifo(j);
                file3 << s;
                if (j < 7)
                {
                    file3 << " ";
                }
                else if (j == 6)
                {
                    file3 << endl;
                }
            }
        }
        file3.close();
        break;
    }
    case 4:
        cout << "返回" << endl;
        break;
    default:
        cout << "错误，已返回" << endl;
        break;
    }
}

void update(string _uid)
{
    fstream file1, file2, file3, file;
    string uid, name, age, salesnum, dep, title;
    int flag = 0;
    file1.open(sales, ios::in | ios::out);
    file.open(text1, ios::out | ios::app);
    if (!file1)
    {
        cout << "Salesman.txt cannot be openned" << endl;
        abort();
    }
    if (!file)
    {
        cout << "1.txt cannot be openned" << endl;
        abort();
    }
    while (1)
    {
        file1 >> uid;
        file1 >> name;
        file1 >> age;
        file1 >> salesnum;
        if (file1.eof())
            break;
        if (_uid == uid && flag == 0)
        {
            cout << "原信息:" << endl;
            cout << "编号：" << uid << endl;
            cout << "姓名：" << name << endl;
            cout << "年龄：" << age << endl;
            cout << "销售额：" << salesnum << endl;
            cout << "请输入新的信息" << endl;
            cout << "编号：" << endl;
            cin >> uid;
            cout << "姓名：" << endl;
            cin >> name;
            cout << "年龄：" << endl;
            cin >> age;
            cout << "销售额：" << endl;
            cin >> salesnum;
            flag = 1;
        }
        file << uid << " ";
        file << name << " ";
        file << age << " ";
        file << salesnum << endl;
    }
    file.close();
    file1.close();
    if (flag == 1)
    {
        remove(sales);
        rename(text1, sales);
        return;
    }
    if (flag == 0)
    {
        remove(text1);
        file2.open(mana, ios::in | ios::out);
        file.open(text1, ios::out | ios::app);
        if (!file2)
        {
            cout << "manager.txt cannot be openned" << endl;
            abort();
        }
        if (!file)
        {
            cout << "1.txt cannot be openned" << endl;
            abort();
        }
        while (1)
        {
            file2 >> uid;
            file2 >> name;
            file2 >> age;
            file2 >> dep;
            if (file2.eof())
                break;
            if (_uid == uid && flag == 0)
            {
                cout << "原信息:" << endl;
                cout << "编号：" << uid << endl;
                cout << "姓名：" << name << endl;
                cout << "年龄：" << age << endl;
                cout << "部门：" << dep << endl;
                cout << "请输入新的信息" << endl;
                cout << "编号：" << endl;
                cin >> uid;
                cout << "姓名：" << endl;
                cin >> name;
                cout << "年龄：" << endl;
                cin >> age;
                cout << "部门：" << endl;
                cin >> dep;
                flag = 1;
            }
            file << uid << "\t";
            file << name << "\t";
            file << age << "\t";
            file << dep << "\t";
            file << endl;
        }
    }
    file.close();
    file2.close();
    if (flag == 1)
    {
        remove(smana);
        rename(text1, smana);
        return;
    }
    if (flag == 0)
    {
        remove(text1);
        file3.open(smana, ios::in | ios::out);
        file.open(text1, ios::out | ios::app);
        if (!file3)
        {
            cout << "salemanager.txt cannot be openned" << endl;
            abort();
        }
        if (!file)
        {
            cout << "1.txt cannot be openned" << endl;
            abort();
        }
        while (1)
        {

            file3 >> uid;
            file3 >> name;
            file3 >> age;
            file3 >> salesnum;
            file3 >> dep;
            file3 >> title;
            if (file3.eof())
                break;
            if (_uid == uid && flag == 0)
            {
                cout << "原信息:" << endl;
                cout << "编号：" << uid << endl;
                cout << "姓名：" << name << endl;
                cout << "年龄：" << age << endl;
                cout << "销售额：" << salesnum << endl;
                cout << "部门：" << dep << endl;
                cout << "职称：" << title << endl;
                cout << "请输入新的信息" << endl;
                cout << "编号：" << endl;
                cin >> uid;
                cout << "姓名：" << endl;
                cin >> name;
                cout << "年龄：" << endl;
                cin >> age;
                cout << "销售额：" << endl;
                cin >> salesnum;
                cout << "部门：" << endl;
                cin >> dep;
                cout << "职称：" << endl;
                cin >> title;
                flag = 1;
            }
            file << uid << "\t";
            file << name << "\t";
            file << age << "\t";
            file << salesnum << "\t";
            file << dep << "\t";
            file << title << endl;
        }
        file.close();
        file3.close();
    }
    if (flag == 1)
    {
        remove(smana);
        rename(text1, smana);
        return;
    }
    if (flag == 0)
    {
        cout << "没有记录！" << endl;
        remove(text1);
    }
}

void look(string _uid)
{
    fstream file1, file2, file3;
    string uid, name, age, salesnum, dep, title;
    int flag = 0;
    file1.open(sales, ios::in);
    if (!file1)
    {
        cout << "打开错误或者文件不存在";
    }
    else
    {
        while (1)
        {
            file1 >> uid;
            file1 >> name;
            file1 >> age;
            file1 >> salesnum;
            if (_uid == uid && flag == 0)
            {
                cout << "编号：" << uid << endl;
                cout << "姓名：" << name << endl;
                cout << "年龄：" << age << endl;
                cout << "销售额：" << salesnum << endl;
                flag = 1;
                break;
            }
            if (file1.eof())
                break;
        }
    }
    file1.close();
    if (flag == 0)
    {
        file2.open(mana, ios::in);
        if (!file2)
        {
            cout << "打开错误或者文件不存在";
        }
        else
        {
            while (1)
            {

                file2 >> uid;
                file2 >> name;
                file2 >> age;
                file2 >> dep;
                if (_uid == uid)
                {
                    cout << "编号：" << uid << endl;
                    cout << "姓名：" << name << endl;
                    cout << "年龄：" << age << endl;
                    cout << "部门：" << dep << endl;
                    flag = 1;
                    break;
                }
                if (file2.eof())
                    break;
            }
        }
    }
    if (flag == 0)
    {
        file3.open(smana, ios::in);
        if (!file3)
        {
            cout << "打开错误或者文件不存在";
        }
        else
        {
            while (1)
            {

                file3 >> uid;
                file3 >> name;
                file3 >> age;
                file3 >> dep;
                if (_uid == uid)
                {
                    cout << "编号：" << uid << endl;
                    cout << "姓名：" << name << endl;
                    cout << "年龄：" << age << endl;
                    cout << "销售额：" << salesnum << endl;
                    cout << "部门：" << dep << endl;
                    cout << "职称：" << title << endl;
                    flag = 1;
                    break;
                }
                if (file3.eof())
                    break;
            }
        }
    }
    if (flag == 0)
        cout << "没有找到这个编号的记录" << endl;
}

void _delete(string _uid)
{
    fstream file1, file2, file3, file4, file;
    string uid, name, age, salesnum, dep, title;
    int flag = 0;
    file1.open(sales, ios::in | ios::out);
    if (!file1)
    {
        cout << "salesman 打开失败" << endl;
        abort();
    }
    file.open(text1, ios::out | ios::app);
    if (!file)
    {
        cout << "salesman 打开失败" << endl;
        abort();
    }
    file4.open(text2, ios::out | ios::app);
    if (!file4)
    {
        cout << "salesman 打开失败" << endl;
        abort();
    }
    while (1)
    {

        file1 >> uid;
        file1 >> name;
        file1 >> age;
        file1 >> salesnum;
        if (file1.eof())
            break;
        if (uid == _uid && flag == 0)
        {
            flag = 1;
            file4 << uid << "\t";
            file4 << name << "\t";
            file4 << age << "\t";
            file4 << salesnum << "\t";
            file4 << "salesman";
            file4 << endl;
            continue;
        }
        file << uid << "\t";
        file << name << "\t";
        file << age << "\t";
        file << salesnum;
        file << endl;
    }
    file.close();
    file1.close();
    if (flag == 1)
    {
        remove(sales);
        rename(text1, sales);
        return;
    }
    if (flag == 0)
    {
        remove(text1);
        file2.open(mana, ios::in | ios::out);
        file.open(text1, ios::out | ios::app);
        if (!file2)
        {
            cout << "manager 打开失败" << endl;
            abort();
        }
        if (!file)
        {
            cout << "manager 打开失败" << endl;
            abort();
        }
        while (1)
        {

            file2 >> uid;
            file2 >> name;
            file2 >> age;
            file2 >> dep;
            if (file2.eof())
                break;
            if (uid == _uid && flag == 0)
            {
                flag = 1;
                file4 << uid << "\t";
                file4 << name << "\t";
                file4 << age << "\t";
                file4 << dep << "\t";
                file4 << "manager";
                file4 << endl;
                continue;
            }
            file << uid << "\t";
            file << name << "\t";
            file << age << "\t";
            file << dep;
            file << endl;
        }
        file.close();
        file2.close();
        if (flag == 1)
        {
            remove(mana);
            rename(text1, mana);
            return;
        }
    }
    if (flag == 0)
    {
        remove(text1);
        file3.open(smana, ios::in | ios::out);
        file.open(text1, ios::out | ios::app);
        if (!file3)
        {
            cout << "salemanager.txt cannot be openned" << endl;
            abort();
        }
        if (!file)
        {
            cout << "1.txt cannot be openned" << endl;
            abort();
        }
        while (1)
        {

            file3 >> uid;
            file3 >> name;
            file3 >> age;
            file3 >> salesnum;
            file3 >> dep;
            file3 >> title;
            if (file3.eof())
                break;
            if (_uid == uid && flag == 0)
            {
                flag = 1;
                file4 << uid << "\t";
                file4 << name << "\t";
                file4 << age << "\t";
                file4 << salesnum << "\t";
                file4 << dep << "\t";
                file4 << title << "\t";
                file4 << "salemanager" << endl;
                continue;
            }
            file << uid << "\t";
            file << name << "\t";
            file << age << "\t";
            file << salesnum << "\t";
            file << dep << "\t";
            file << title << endl;
        }
        file.close();
        file4.close();
    }
    if (flag == 1)
    {
        remove(smana);
        rename(text1, smana);
        return;
    }
    if (flag == 0)
    {
        cout << "没有记录！" << endl;
        remove(text1);
    }
    file4.close();
}

int main()
{
    cout << "学号:17030140076  姓名:雷鸣宇   班级:1703014" << endl;
    cout << "请选择操作：" << endl;
    cout << "1：增加数据" << endl;
    cout << "2：更新数据" << endl;
    cout << "3：查询数据" << endl;
    cout << "4：删除数据进入回收站" << endl;
    cout << "5：清空回收站数据" << endl;
    cout << "6：退出系统" << endl;
    int i;
    string s;
    while (cin >> i)
    {
        switch (i)
        {
        case 1:
            add();
            break;
        case 2:
            cout << "请输入要更新的uid" << endl;
            cin >> s;
            update(s);
            break;
        case 3:
            cout << "请输入要查询的uid" << endl;
            cin >> s;
            look(s);
            break;
        case 4:
            cout << "请输入要删除的uid" << endl;
            cin >> s;
            _delete(s);
            break;
        case 5:
            remove(text2);
            break;
        case 6:
            return 0;
        default:
            cout << "输入错误，请重新输入";
            break;
        }
        cout << "学号:17030140076  姓名:雷鸣宇   班级:1703014" << endl;
        cout << "请选择操作：" << endl;
        cout << "1：增加数据" << endl;
        cout << "2：更新数据" << endl;
        cout << "3：查询数据" << endl;
        cout << "4：删除数据进入回收站" << endl;
        cout << "5：清空回收站数据" << endl;
        cout << "6：退出系统" << endl;
    }
}
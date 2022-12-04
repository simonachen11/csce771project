#include<iostream>
#include <fstream>
#include <string>
#include <sstream>
#include<map>
#include<vector>

using namespace std;
 

void solve() {
    for (int i = 2000; i <= 2014; i++) {
        string path = "nibrs_criminal_act_";
        path += to_string(i);
        path += ".csv";
        ifstream fp(path);
        string line;
        getline(fp, line);

        map<int, int>mp;
        while (getline(fp, line)) { //循环读取每行数据
            vector<string> data_line;
            string number;
            istringstream readstr(line); //string数据流化
            //将一行数据按'，'分割
            for (int j = 0; j < 13; j++) { //可根据数据的实际情况取循环获取
                getline(readstr, number, ','); //循环读取数据
                data_line.push_back(number); //字符串传int
            }

            mp[atoi(data_line[0].c_str())]++;
        }
        for (auto x : mp) {
            cout << x.first << " " << x.second << endl;
        }
    }
     
    
     
}
int main() {
	solve(); 


	return 0; 
}
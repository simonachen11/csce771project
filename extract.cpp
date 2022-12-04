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
        while (getline(fp, line)) { //ѭ����ȡÿ������
            vector<string> data_line;
            string number;
            istringstream readstr(line); //string��������
            //��һ�����ݰ�'��'�ָ�
            for (int j = 0; j < 13; j++) { //�ɸ������ݵ�ʵ�����ȡѭ����ȡ
                getline(readstr, number, ','); //ѭ����ȡ����
                data_line.push_back(number); //�ַ�����int
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
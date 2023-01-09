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
        while (getline(fp, line)) { 
            vector<string> data_line;
            string number;
            istringstream readstr(line); 
            //slit every line by '，'
            for (int j = 0; j < 13; j++) { 
                getline(readstr, number, ','); 
                data_line.push_back(number); 
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

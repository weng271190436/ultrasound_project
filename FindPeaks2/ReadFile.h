#include <iostream>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>

using namespace boost::filesystem;
using namespace std;

void get_all(const path& root, const string& ext, vector<path>& ret);
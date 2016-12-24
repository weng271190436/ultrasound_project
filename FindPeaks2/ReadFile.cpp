#include "ReadFile.h"

using namespace boost::filesystem;
using namespace std;

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const path& root, const string& ext, vector<path>& ret)
{
    if(!exists(root) || !is_directory(root)) return;

    recursive_directory_iterator it(root);
    recursive_directory_iterator endit;

    while(it != endit)
    {
        if(is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path().filename());
        ++it;

    }

}
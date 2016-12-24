#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

template <typename T>
std::string to_string(const T& object);

void convert_image(string filepath, string filename, string dst_path,
	vector< vector<int> > & arrays, vector<bool> & peaks, int & lastPeakPos, 
	int index);
void find_peak(vector< vector<int> > & arrays, int index, vector<int> & peaks, 
	int & lastPeakPos );
void output(vector<int> & peaks_index, int numOfFiles, string path);
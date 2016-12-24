#include "ReadFile.h"
#include "Main.h"

using namespace boost::filesystem;
using namespace std;
using namespace cv;

template <typename T>
string to_string(const T& object) {
    std::ostringstream ss;
    ss << object;
    return ss.str();
}

void convert_image(string filepath, string filename, string dst_path, vector< vector<int> > & arrays, int index) {
    //vector<int> * ans;

    //cout << (arrays&) << endl;

	// load the source image
    Mat src = imread( filepath, IMREAD_COLOR );

    //Mat dst = src.clone();
    
    // accept only char type matrices
    //CV_Assert(dst.depth() == CV_8U);
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols;

    int row, col;
    for ( col = 0; col < nCols; ++col ) {
    	int minRowWithGreenPixel = 600;
        for( row = 500; row < 551; ++row ) {
            Vec3b pixel = src.at<Vec3b>(row, col);
            
            if (pixel[1] - pixel[0] >= 28 && pixel[1] - pixel[2] >= 28 && row < minRowWithGreenPixel){
                //pixel[1] = 0;
                //cout << row << endl;
                minRowWithGreenPixel = row;
                //ans.push_back(600 - row);
            } //else {
                //ans.push_back(0);
            //}

            //if (row < 400) pixel[1] = 0;

            //pixel[0] = 0;
            //pixel[2] = 0;

            //dst.at<Vec3b>(row,col) = pixel;
        }
        //cout << minRowWithGreenPixel << endl;
        arrays[index][col] = 600 - minRowWithGreenPixel;
    }


    //Mat final;
    //Mat aux = dst.colRange(0,800).rowRange(500,600);
    //aux.copyTo(final);
    
    //vector<int> compression_params;
    //string new_path = dst_path + "/" + filename;

    //imwrite(new_path, final, compression_params);
}

void find_peak(vector< vector<int> > & arrays, int index, vector<int> & peaks, int & lastPeakPos ) {
	/*int currPos = lastPeakPos + 10;
	int maxPosMoved = 250;
	int posMoved = 0;

	// 788 is the last col the ultrasound graph can reach
	while (currPos < 789 && posMoved < maxPosMoved 
		&& arrays[i][currPos] < 50) 
	{
		currPos++;
		posMoved++;
	}

	if (arrays[i][currPos] >= 50) lastPeakPos = currPos;
	// 
	else if (currPos >= 789) {
		while (currPos < 789 && posMoved < maxPosMoved 
		    && arrays[i][currPos] < 50) 
	    {
		    currPos++;
		    posMoved++;
	    }
	    if (arrays[i][currPos] >= 50) lastPeakPos = currPos;
	}*/

	bool peak = false;

	int frontier = ((int)(index/244.0*722))%722 + 66;
    //cout << "index: " << index << endl;
    //cout << "frontier: " <<frontier << endl;

	for (int i = -3; i < 4; i++) {
		int pos = frontier + i;
		if (pos < 66) pos+=722;
		else if (pos >= 788) pos-=722;

		//cout << arrays[index][pos] << " ";

		if (arrays[index][pos] >= 50) peak = true;
	}
	//cout << endl;

	if (peak == true) {
		if (peaks.size() == 0)
		    peaks.push_back(index);
		else if (peaks.size() > 0 && index - peaks[peaks.size()-1] > 5)
			peaks.push_back(index);
	}
}

void output(vector<int> & peaks_index, int numOfFiles, string path) {
    // open or create textfile
    ofstream myfile;
    myfile.open (path + "/output.txt", fstream::out);

    //cout << "hey" << endl;
    int counter = 0;

    int curr_idx = 0;
    for (; curr_idx < peaks_index[0]; curr_idx++) {
    	myfile << "-1" << " ";
    	counter++;
    }

    // output 0 for peaks_index[0]
    //myfile << 0 << " ";
    curr_idx = peaks_index[0] + 1;

    for (int i = 1; i < peaks_index.size(); i++) {
    	double diff = peaks_index[i] - curr_idx;
    	for (;curr_idx < peaks_index[i]; curr_idx++) {
    		myfile << 1- (peaks_index[i] - curr_idx)/diff << " ";
    		counter++;
     	}
     	//curr_idx = peaks_index[i] + 1;
    }

    myfile << 0 << " ";
    counter++;
    //curr_idx++;

    for (; curr_idx < numOfFiles; curr_idx++) {
    	myfile << "-1" << " ";
    	counter++;
    }
    //cout << "curr_idx: " << curr_idx << endl; 
    //cout << "counter: " << counter << endl;
	myfile << endl;
	myfile.close();
}

int main( int argc, char** argv ) {
    
    
	path p = current_path();
    p += "/image_src/";
	directory_iterator it(p);
	while (it != directory_iterator()) {
	  	if (is_directory(*it)){
	  		cout << *it << '\n';
	  		// Get Images
	  		vector<path> files; // image file paths
	  		
		  	string img_src = to_string(*it);
		  	img_src = img_src.substr(1, img_src.length()-2);
		    path root (img_src);
			string ext = ".jpg";
			get_all(root, ext, files);

		    vector< vector<int> > arrays(files.size(), vector<int>(800)); // array representation of images
		    vector<int> peaks; // positions of peaks
		    //vector<bool> ans(files.size());
		    cout << files.size() << endl;

			for (int i = 0; i < files.size(); i++) {
				//cout << i << " " << flush;
				string filename = to_string(files[i]);
				filename = filename.substr(1, filename.length()-2);
				string filepath = img_src + "/" + filename;
				convert_image(filepath, filename, "image_dst", arrays, i);

			}

			//cout << " hey" << flush;
			//cout << arrays[0].size() << endl;
		    int lastPeakPos = 0;
			for (int i = 0; i < files.size(); i++) {
				find_peak(arrays, i, peaks, lastPeakPos);
				// cout << endl;
				// for (int j = 0; j < arrays[i].size(); j++) {
		  //           cout << arrays[i][j] << " ";
				// }
				// cout << endl;
			}
			//cout << peaks.size() << flush;

			for (int i = 0; i < peaks.size(); i++) {
				cout << peaks[i] << " ";
			}
			cout << endl;
			//cout << " hey" << flush;

		    output(peaks, files.size(), img_src);
	    }
        
	    it++;
	}
	
	return 0;
}
#include <pthread.h>
#include <Eigen/Dense>
#include <bitset>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define FACTOR 1
#define MAX_BITSET_SIZE (1000000 / FACTOR)
#define SEARCH_RADIUS 500
#define PI 3.14159265
#define BUFFER 20.0

using Eigen::Vector2d;
using Eigen::Vector3d;

using std::bitset;
using std::cerr;
using std::cout;
using std::endl;
using std::flush;
using std::ifstream;
using std::make_pair;
using std::max;
using std::min;
using std::ofstream;
using std::pair;
using std::string;
using std::vector;
using boost::filesystem::path;
using boost::filesystem::recursive_directory_iterator;

//string[6] colors = {"red", "yellow", "blue", "green", "pink"};

struct Line {
  Vector2d start;
  Vector2d end;
  Vector2d origin;
  double offset;
  double theta;
  int half_width;
  int half_length; 
};

struct Result {
  Line best_line0;
  Line best_line1;
};

struct Column {
  int red[6];
  int yellow[6];
  int blue[6];
  int pink[6];
  int green[6];
  int red_size = 0;
  int yellow_size = 0;
  int blue_size = 0;
  int pink_size = 0;
  int green_size = 0;
};

struct Columns {
  Column cols[500];
  int cols_size = 0;
};

const int kNumThreads = 4;
Result global_results[kNumThreads];

struct SpatialTemporalImageArguments {

};

template <typename T>
string to_string(const T& object) {
  std::ostringstream ss;
  ss << object;
  return ss.str();
}

double distance(Vector2d p1, Vector2d p2) {
  return sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]));
}

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

void KeepRedPart(cv::Mat& src, cv::Mat& output) {
  output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = output.at<cv::Vec3b>(row, col);
      if (pixel[0] < 30 && pixel[1] < 30 && pixel[2] > 230) {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 255;
      }
      else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
}

bool isRed(cv::Vec3b pixel) {
  if (pixel[0] < 30 && pixel[1] < 30 && pixel[2] > 230) {
    return true;
  } else {
    return false;
  }
}

void KeepGreenPart(cv::Mat& src, cv::Mat& output) {
  output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = output.at<cv::Vec3b>(row, col);
      if (pixel[0] < 100 && pixel[1] > 90 && pixel[1] < 140 && pixel[2] < 30) {
        pixel[0] = 0;
        pixel[1] = 255;
        pixel[2] = 0;
      }
      else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
}

bool isGreen(cv::Vec3b pixel) {
  if (pixel[0] < 30 && pixel[1] > 230 && pixel[2] < 30) {
    return true;
  } else {
    return false;
  }
}

void KeepBluePart(cv::Mat& src, cv::Mat& output) {
  output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = output.at<cv::Vec3b>(row, col);
      if (pixel[0] > 230 && pixel[1] < 30 && pixel[2] < 30) {
        pixel[0] = 255;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
}

bool isBlue(cv::Vec3b pixel) {
  if (pixel[0] > 230 && pixel[1] < 30 && pixel[2] < 30) {
    return true;
  } else {
    return false;
  }
}

void KeepYellowPart(cv::Mat& src, cv::Mat& output) {
  output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = output.at<cv::Vec3b>(row, col);
      if (pixel[0] < 60 && pixel[1] > 220 && pixel[2] > 220) {
        pixel[0] = 0;
        pixel[1] = 255;
        pixel[2] = 255;
      }
      else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
}

bool isYellow(cv::Vec3b pixel) {
  if (pixel[0] < 60 && pixel[1] > 220 && pixel[2] > 220) {
    return true;
  } else {
    return false;
  }
}

void KeepPinkPart(cv::Mat& src, cv::Mat& output) {
  output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = output.at<cv::Vec3b>(row, col);
      if (pixel[0] > 130 && pixel[0] < 220 && pixel[1] > 90 && pixel[1] < 140 && pixel[2] > 230) {
        pixel[0] = 216;
        pixel[1] = 138;
        pixel[2] = 255;
      }
      else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
}

bool isPink(cv::Vec3b pixel) {
  if (pixel[0] > 130 && pixel[0] < 220 && pixel[1] > 90 && pixel[1] < 140 && pixel[2] > 230) {
    return true;
  } else {
    return false;
  }
}

void ToBlackAndWhiteImages(cv::Mat& src, cv::Mat& output) {
	output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = output.at<cv::Vec3b>(row, col);
      if (pixel[0] < 100 && pixel[1] < 100 && pixel[2] < 100) {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      else {
      	pixel[0] = 255;
      	pixel[1] = 255;
      	pixel[2] = 255;
      }
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
}

bool ToBitSet(cv::Mat& src, bitset<MAX_BITSET_SIZE>& output) {
  if (src.rows * src.cols > output.size())
  	return false;
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);
      if (pixel[0] < 100 && pixel[1] < 100 && pixel[2] < 100) {
        output.set(col + row * nCols, 0);
      }
      else {
      	output.set(col + row * nCols, 1);
      }
    }
  }
  return true;
}

bool CropImage(int first_row, int last_row, cv::Mat& src, cv::Mat& output) {
  if (first_row < 0 || first_row >= src.rows)
  	return false;
  if (last_row < 0 || last_row >= src.rows)
  	return false;
  if (last_row < first_row)
  	return false;

  output = src.clone();
  int nRows = src.rows;
  int nCols = src.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < first_row; ++row ) {
      cv::Vec3b pixel(0, 0, 0);
      output.at<cv::Vec3b>(row, col) = pixel;
    }
    for( row = last_row+1; row < nRows; ++row ) {
      cv::Vec3b pixel(0, 0, 0);
      output.at<cv::Vec3b>(row, col) = pixel;
    }
  }
  return true;
}

pair<double, double> FastComputeScore(const bitset<MAX_BITSET_SIZE>& signal0, 
	                                    const bitset<MAX_BITSET_SIZE>& signal1,
	                                    int width) {
  // Check slide offset.
  int best_score = 0;
  int best_offset = 0;
  for (int offset = -SEARCH_RADIUS; offset <= SEARCH_RADIUS; ++offset) {
    bitset<MAX_BITSET_SIZE> signal1_shifted;
    if (offset < 0)
      signal1_shifted = (signal1 >> (offset * width) );
    else if (offset > 0)
      signal1_shifted = (signal1 << (offset * width) );

    const int score = (signal0 & signal1_shifted).count();
    if (score > best_score) {
      best_score = score;
      best_offset = offset;
    }
  }

  const unsigned long kMinimumCount = 4;
  const int denom = max(kMinimumCount, (signal0 | signal1).count());
  return make_pair<double, double>(best_score / static_cast<double>(denom), best_offset);
}

bool ShiftImage(cv::Mat src, cv::Mat output, int offset) {
	if (src.rows != output.rows || src.cols != output.cols) return false;
	int nRows = output.rows;
  int nCols = output.cols;

  int output_row, output_col;
  int src_row, src_col;
  for ( output_row = 0; output_row < nRows; ++output_row ) {

    src_row = output_row - offset;
    if (src_row >= 0 && src_row < nRows){
    	for( output_col = 0; output_col < nCols; ++output_col ) {
	      cv::Vec3b pixel = src.at<cv::Vec3b>(src_row, output_col);
	      output.at<cv::Vec3b>(output_row, output_col) = pixel;
	    }
    } else {
      for( output_col = 0; output_col < nCols; ++output_col ) {
	      cv::Vec3b pixel;
	      pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
	      output.at<cv::Vec3b>(output_row, output_col) = pixel;
	    }
    }
    
  }
  return true;
}

pair<double, double> ReadImageAndComputeScore(cv::Mat& src0, cv::Mat& src1) {
	cv::Mat croppedSrc0(src0);
  cv::Mat croppedSrc1(src1);
  // CropImage(275, 350, src0 ,croppedSrc0);
  // CropImage(275, 350, src1 ,croppedSrc1);

  cv::Mat output0;
  cv::Mat output1;
  int width = src0.cols;
  bitset<MAX_BITSET_SIZE> signal0;
  bitset<MAX_BITSET_SIZE> signal1;
  ToBlackAndWhiteImages(croppedSrc0, output0);
  ToBlackAndWhiteImages(croppedSrc1, output1);
  ToBitSet(output0, signal0);
  ToBitSet(output1, signal1);
  pair<double, double> ans = FastComputeScore(signal0, signal1, width);
  //cv::Mat shiftedOutput1 = cv::Mat::zeros(output1.rows, output1.cols, CV_8UC3);
  //ShiftImage(output1, shiftedOutput1, 5);

  // cv::namedWindow( "Display window0", cv::WINDOW_AUTOSIZE );
  // cv::imshow( "Display window0", src0 ); 
  // cv::namedWindow( "Display window1", cv::WINDOW_AUTOSIZE );
  // cv::imshow( "Display window1", src1 );   
  // cv::waitKey(0);
  
  return ans;
}

bool ReadImages(string path0, string path1,
                vector<cv::Mat>& images0,
                vector<cv::Mat>& images1) {

  vector<path> files; // image file paths
  path root (path0);

  string ext = ".jpg";
  get_all(root, ext, files);

  std::vector<std::string> filepaths;
  filepaths.resize(files.size());

  for (int i = 0; i < files.size(); i++) {
    string filename = to_string(files[i]);
    filename = filename.substr(1, filename.length()-2);
    string filepath = path0 + "/" + filename;
    filepaths[i] = filepath;
  }
  for (int i = 0; i < filepaths.size(); i++) {
    // load the source image
    cv::Mat src = cv::imread( filepaths[i], CV_LOAD_IMAGE_COLOR );
    //cv::Mat resize_src(src.rows/FACTOR, src.cols/FACTOR, CV_LOAD_IMAGE_COLOR);
    //resize(src, resize_src, resize_src.size(), 0, 0, cv::INTER_LINEAR);
    images0.push_back(src);
    // cv::namedWindow( "Display window0", cv::WINDOW_AUTOSIZE );
    // cv::imshow( "Display window0", resize_src );
    // cv::waitKey(0);
  }

  files.clear(); // image file paths
  root = path(path1);

  ext = ".jpg";
  get_all(root, ext, files);

  filepaths.clear();
  filepaths.resize(files.size());

  for (int i = 0; i < files.size(); i++) {
    string filename = to_string(files[i]);
    filename = filename.substr(1, filename.length()-2);
    string filepath = path1 + "/" + filename;
    filepaths[i] = filepath;
  }
  for (int i = 0; i < filepaths.size(); i++) {
    // load the source image
    cv::Mat src = cv::imread( filepaths[i], CV_LOAD_IMAGE_COLOR );
    // cv::Mat resize_src(src.rows/FACTOR, src.cols/FACTOR, CV_LOAD_IMAGE_COLOR);
    // resize(src, resize_src, resize_src.size(), 0, 0, cv::INTER_LINEAR);
    images1.push_back(src);
  }

  return true;
}

bool ResizeImages(vector<cv::Mat>& src0, vector<cv::Mat>& src1,
                  vector<cv::Mat>& dst0, vector<cv::Mat>& dst1) {
  for (int i = 0; i < src0.size(); i++){
    cv::Mat src = src0[i];
    cv::Mat resize_src(src.rows/FACTOR, src.cols/FACTOR, CV_LOAD_IMAGE_COLOR);
    resize(src, resize_src, resize_src.size(), 0, 0, cv::INTER_LINEAR);
    dst0.push_back(resize_src);
  }

  for (int i = 0; i < src1.size(); i++){
    cv::Mat src = src1[i];
    cv::Mat resize_src(src.rows/FACTOR, src.cols/FACTOR, CV_LOAD_IMAGE_COLOR);
    resize(src, resize_src, resize_src.size(), 0, 0, cv::INTER_LINEAR);
    dst1.push_back(resize_src);
  }
  return true;
}

void MakeColumn(Line& line, cv::Mat& src , Column& column) {
  int channels = src.channels();
  int nRows = src.rows;
  int nCols = src.cols;
  int half_length = line.half_length;
  int half_width = line.half_width;
  Vector2d start = line.start;
  Vector2d end = line.end;
  Vector2d origin = line.origin;
  double offset = line.offset;
  double theta = line.theta;

  cv::Vec3b last_pixel = cv::Vec3b(0, 0, 0);

  if ((theta > PI/4 && theta < PI/4*3) || (theta > PI/4*5 && theta < PI/4*7)) {
    int index = 0;
    for (int col = origin[0] - half_length; col < origin[0] + half_length; col++) {
      int row = (offset - cos(theta) * (col-origin[0]))/sin(theta);
      row+=origin[1];
      cv::Vec3b color = src.at<cv::Vec3b>(row, col);
      if (isRed(color) && !isRed(last_pixel) && column.red_size % 2 == 0) {
        if (column.red_size < 6) {
          column.red[column.red_size] = index;
          column.red_size++;
        }

      } else if (isRed(last_pixel) &&  !isRed(color) && column.red_size % 2 == 1) {
        if (column.red_size < 6) {
          column.red[column.red_size] = index;
          column.red_size++;
        }
      }

      if (isYellow(color) && !isYellow(last_pixel) && column.yellow_size % 2 == 0) {
        if (column.yellow_size < 6) {
          column.yellow[column.yellow_size] = index;
          column.yellow_size++;
        }

      } else if (isYellow(last_pixel) &&  !isYellow(color) && column.yellow_size % 2 == 1) {
        if (column.yellow_size < 6) {
          column.yellow[column.yellow_size] = index;
          column.yellow_size++;
        }
      }

      if (isBlue(color) && !isBlue(last_pixel) && column.blue_size % 2 == 0) {
        if (column.blue_size < 6) {
          column.blue[column.blue_size] = index;
          column.blue_size++;
        }

      } else if (isBlue(last_pixel) &&  !isBlue(color) && column.blue_size % 2 == 1) {
        if (column.blue_size < 6) {
          column.blue[column.blue_size] = index;
          column.blue_size++;
        }
      }

      if (isPink(color) && !isPink(last_pixel) && column.pink_size % 2 == 0) {
        if (column.pink_size < 6) {
          column.pink[column.pink_size] = index;
          column.pink_size++;
        }

      } else if (isPink(last_pixel) &&  !isPink(color) && column.pink_size % 2 == 1) {
        if (column.pink_size < 6) {
          column.pink[column.pink_size] = index;
          column.pink_size++;
        }
      }

      if (isGreen(color) && !isGreen(last_pixel) && column.green_size % 2 == 0) {
        if (column.green_size < 6) {
          column.green[column.green_size] = index;
          column.green_size++;
        }

      } else if (isGreen(last_pixel) &&  !isGreen(color) && column.green_size % 2 == 1) {
        if (column.green_size < 6) {
          column.green[column.green_size] = index;
          column.green_size++;
        }
      }

      last_pixel = color;
      index++;
    }
    
  } else {
    int index = 0;
    for (int row = origin[1] - half_length; row < origin[1] + half_length; row++) {
      int col = (offset - sin(theta) * (row-origin[1]))/cos(theta);
      col+=origin[0];
      cv::Vec3b color = src.at<cv::Vec3b>(row, col);
      if (isRed(color) && !isRed(last_pixel) && column.red_size % 2 == 0) {
        if (column.red_size < 6) {
          column.red[column.red_size] = index;
          column.red_size++;
        }

      } else if (isRed(last_pixel) &&  !isRed(color) && column.red_size % 2 == 1) {
        if (column.red_size < 6) {
          column.red[column.red_size] = index;
          column.red_size++;
        }
      }

      if (isYellow(color) && !isYellow(last_pixel) && column.yellow_size % 2 == 0) {
        if (column.yellow_size < 6) {
          column.yellow[column.yellow_size] = index;
          column.yellow_size++;
        }

      } else if (isYellow(last_pixel) &&  !isYellow(color) && column.yellow_size % 2 == 1) {
        if (column.yellow_size < 6) {
          column.yellow[column.yellow_size] = index;
          column.yellow_size++;
        }
      }

      if (isBlue(color) && !isBlue(last_pixel) && column.blue_size % 2 == 0) {
        if (column.blue_size < 6) {
          column.blue[column.blue_size] = index;
          column.blue_size++;
        }

      } else if (isBlue(last_pixel) &&  !isBlue(color) && column.blue_size % 2 == 1) {
        if (column.blue_size < 6) {
          column.blue[column.blue_size] = index;
          column.blue_size++;
        }
      }

      if (isPink(color) && !isPink(last_pixel) && column.pink_size % 2 == 0) {
        if (column.pink_size < 6) {
          column.pink[column.pink_size] = index;
          column.pink_size++;
        }

      } else if (isPink(last_pixel) &&  !isPink(color) && column.pink_size % 2 == 1) {
        if (column.pink_size < 6) {
          column.pink[column.pink_size] = index;
          column.pink_size++;
        }
      }

      if (isGreen(color) && !isGreen(last_pixel) && column.green_size % 2 == 0) {
        if (column.green_size < 6) {
          column.green[column.green_size] = index;
          column.green_size++;
        }

      } else if (isGreen(last_pixel) &&  !isGreen(color) && column.green_size % 2 == 1) {
        if (column.green_size < 6) {
          column.green[column.green_size] = index;
          column.green_size++;
        }
      }

      last_pixel = color;
      index++;
    }
  }
}

pair<int, int> FindClosestLeft(int integer, std::vector<int> vec) {
  int min_diff = 10000;
  int min_index = 0;
  for (int i = 0; i < vec.size(); i+=2) {
    if (abs(vec[i] - integer) < min_diff) {
      min_diff = abs(vec[i] - integer);
      min_index = i;
    }
  }
  pair<int, int> ans;
  ans.first = min_diff;
  ans.second = min_index;
  return ans;
}

pair<int, int> FindClosestRight(int integer, std::vector<int> vec) {
  int min_diff = 10000;
  int min_index = 0;
  for (int i = 1; i < vec.size(); i+=2) {
    if (abs(vec[i] - integer) < min_diff) {
      min_diff = abs(vec[i] - integer);
      min_index = i;
    }
  }
  pair<int, int> ans;
  ans.first = min_diff;
  ans.second = min_index;
  return ans;
}

double CalculateScoreForOneColor(int color0[6], int color0_size, int color1[6], int color1_size ) {
  bool continueFlag = true;
  double scores = 0;
  std::vector<int> vec0;
  std::vector<int> vec1;

  for (int i = 0; i < color0_size; i++) {
    vec0.push_back(color0[i]);
  }
  for (int i = 0; i < color1_size; i++) {
    vec1.push_back(color1[i]);
  }
  
  int numOfMatches = 0;
  while (continueFlag && vec0.size() >= 2 && vec1.size() >= 2) {
    int leftDiff = 1000;
    int leftIndex0 = 1000;
    int leftIndex1 = 1000;
    for (int i = 0; i < vec0.size(); i+=2) {
      pair<int, int> closest = FindClosestLeft(vec0[i], vec1);
      int diff = closest.first;
      if (leftDiff > diff) {
        leftDiff = diff;
        leftIndex0 = i;
        leftIndex1 = closest.second;
      }
    }

    int rightDiff = 1000;
    int rightIndex0 = 1000;
    int rightIndex1 = 1000;
    for (int i = 1; i < vec0.size(); i+=2) {
      pair<int, int> closest = FindClosestRight(vec0[i], vec1);
      int diff = closest.first;
      if (rightDiff > diff) {
        rightDiff = diff;
        rightIndex0 = i;
        rightIndex1 = closest.second;
      }
    }
    // cout << leftIndex0 << " " << rightIndex0 << " " << leftIndex1 << " " << rightIndex1 << endl;
    // cout << vec0.size() << " " << vec1.size() << endl;
    vec0.erase(vec0.begin() + leftIndex0);
    vec0.erase(vec0.begin() + rightIndex0 - 1);
    vec1.erase(vec1.begin() + leftIndex1);
    vec1.erase(vec1.begin() + rightIndex1 - 1);

    if (leftDiff + rightDiff > BUFFER) {
      continueFlag = false;
    } else {
      numOfMatches++;
      scores+= (BUFFER - leftDiff - rightDiff)/BUFFER;
    }


  }
  double final_score = scores/numOfMatches;
  if (numOfMatches ==  0) {
    final_score = 0;
  }
  return final_score;
}

void CopyColumn(Column& src, Column& dst) {
  for (int i = 0; i < src.red_size; i++) {
    dst.red[i] = src.red[i];
  }
  for (int i = 0; i < src.yellow_size; i++) {
    dst.yellow[i] = src.yellow[i];
  }
  for (int i = 0; i < src.blue_size; i++) {
    dst.blue[i] = src.blue[i];
  }
  for (int i = 0; i < src.pink_size; i++) {
    dst.pink[i] = src.pink[i];
  }
  for (int i = 0; i < src.green_size; i++) {
    dst.green[i] = src.green[i];
  }

  dst.red_size = src.red_size;
  dst.yellow_size = src.yellow_size;
  dst.blue_size = src.blue_size;
  dst.pink_size = src.pink_size;
  dst.green_size = src.green_size;

}

void AddOffsetToColumn(Column& col, int offset) {
  for (int i = 0; i < col.red_size; i++) {
    col.red[i] += offset;
  }
  for (int i = 0; i < col.yellow_size; i++) {
    col.yellow[i] += offset;
  }
  for (int i = 0; i < col.blue_size; i++) {
    col.blue[i] += offset;
  }
  for (int i = 0; i < col.pink_size; i++) {
    col.pink[i] += offset;
  }
  for (int i = 0; i < col.green_size; i++) {
    col.green[i] += offset;
  }

}

pair<double, int> FindBestScoreAndOffset(Columns& cols0, Columns& cols1) {
  pair<double, int> ans;
  ans.first = 0;
  ans.second = 0;

  return ans;
}

double CalculateScores(Columns& cols0, Columns& cols1, int offset) {
  double score = 0.0;
  for (int i = 0; i < cols0.cols_size; i++) {
    Column col0 = cols0.cols[i];
    Column col1 = cols1.cols[i];
    Column new_col1;
    CopyColumn(col1, new_col1);
    AddOffsetToColumn(new_col1, offset);
    
    score += CalculateScoreForOneColor(col0.red, col0.red_size, new_col1.red, new_col1.red_size);
    score += CalculateScoreForOneColor(col0.yellow, col0.yellow_size, new_col1.yellow, new_col1.yellow_size);
    score += CalculateScoreForOneColor(col0.blue, col0.blue_size, new_col1.blue, new_col1.blue_size);
    score += CalculateScoreForOneColor(col0.green, col0.green_size, new_col1.green, new_col1.green_size);
    score += CalculateScoreForOneColor(col0.pink, col0.pink_size, new_col1.pink, new_col1.pink_size);
  }
  
  return score;
  
}

pair<int, int> FindOffset(Columns& cols0, Columns& cols1) {
  vector<int> offsets;
  int best_offset;
  int max_score = 0;
  for (int i = 0; i < cols0.cols_size; i++) {
    Column col0 = cols0.cols[i];
    Column col1 = cols1.cols[i];
    vector<int> possible_offsets;
    for (int j = 0; j < col1.red_size; j+=2) {
      for (int k = 0; k < col0.red_size; k+=2) {
        int possible_offset = col1.red[j] - col0.red[k];
        possible_offsets.push_back(possible_offset);
      }
    }

    for (int j = 0; j < col1.yellow_size; j+=2) {
      for (int k = 0; k < col0.yellow_size; k+=2) {
        int possible_offset = col1.yellow[j] - col0.yellow[k];
        possible_offsets.push_back(possible_offset);
      }
    }

    for (int j = 0; j < col1.blue_size; j+=2) {
      for (int k = 0; k < col0.blue_size; k+=2) {
        int possible_offset = col1.blue[j] - col0.blue[k];
        possible_offsets.push_back(possible_offset);
      }
    }

    for (int j = 0; j < col1.pink_size; j+=2) {
      for (int k = 0; k < col0.pink_size; k+=2) {
        int possible_offset = col1.pink[j] - col0.pink[k];
        possible_offsets.push_back(possible_offset);
      }
    }

    for (int j = 0; j < col1.green_size; j+=2) {
      for (int k = 0; k < col0.green_size; k+=2) {
        int possible_offset = col1.green[j] - col0.green[k];
        possible_offsets.push_back(possible_offset);
      }
    }

    for (int j = 0; j < possible_offsets.size(); j++) {
      // int total_difference = 0;
      double score = 0;


      // for (int k = 0; k < min(col1.red_size, col0.red_size); k++) {
      //   //cout << col1.red[k] - col0.red[k] << endl;
      //   total_difference += abs((col1.red[k] - col0.red[k]) - possible_offsets[j]);
      // }

      score += CalculateScores(cols0, cols1, possible_offsets[j]);

      // for (int k = 0; k < min(col1.yellow_size, col0.yellow_size); k++) {
      //   total_difference += abs((col1.yellow[k] - col0.yellow[k]) - possible_offsets[j]);
      // }

      // for (int k = 0; k < min(col1.blue_size, col0.blue_size); k++) {
      //   total_difference += abs((col1.blue[k] - col0.blue[k]) - possible_offsets[j]);
      // }

      // for (int k = 0; k < min(col1.green_size, col0.green_size); k++) {
      //   total_difference += abs((col1.green[k] - col0.green[k]) - possible_offsets[j]);
      // }

      // for (int k = 0; k < min(col1.pink_size, col0.pink_size); k++) {
      //   total_difference += abs((col1.pink[k] - col0.pink[k]) - possible_offsets[j]);
      // }

      if (max_score < score) {
        max_score = score;
        best_offset = possible_offsets[j];
      }

    }
  }


  pair<int, int> ret = {best_offset, max_score};
  return ret;
}

void PrintColumn(Column & column0) {
  cout << "RED: ";
  for (int i = 0; i < column0.red_size; i++) {
    cout << column0.red[i] << " ";
  }
  cout << endl;

  cout << "YELLOW: ";
  for (int i = 0; i < column0.yellow_size; i++) {
    cout << column0.yellow[i] << " ";
  }
  cout << endl;

  cout << "BLUE: ";
  for (int i = 0; i < column0.blue_size; i++) {
    cout << column0.blue[i] << " ";
  }
  cout << endl;

  cout << "GREEN: ";
  for (int i = 0; i < column0.green_size; i++) {
    cout << column0.green[i] << " ";
  }
  cout << endl;

  cout << "PINK: ";
  for (int i = 0; i < column0.pink_size; i++) {
    cout << column0.pink[i] << " ";
  }
  cout << endl;

}

pair<int, int> MakeColumns(vector<cv::Mat>& src0, vector<cv::Mat>& src1, 
                Columns& output0, Columns& output1, Line& line0, Line& line1) {
  for (int i = 0; i < src0.size(); i++) {
    Column column;
    MakeColumn(line0, src0[i], column);
    //PrintColumn(column);
    output0.cols[i] = column;
    output0.cols_size++; 
    //cout<<endl;
  }
  for (int i = 0; i < src1.size(); i++) {
    Column column;
    MakeColumn(line1, src0[i], column);
    //PrintColumn(column);
    output1.cols[i] = column;
    output1.cols_size++; 
    //cout<<endl;
  }

  //cout<<endl;cout<<endl;

  pair<int, int> offset_and_diff = FindOffset(output0, output1);
  // int offset = offset_and_diff[0];

  // for (int i = 0; i < output1.cols_size; i++) {
  //   for (int j = 0; j < red_size; j++) {
  //     output1.red[j] += offset;
  //   }
  //   for (int j = 0; j < blue_size; j++) {
  //     output1.blue[j] += offset;
  //   }
  //   for (int j = 0; j < yellow_size; j++) {
  //     output1.yellow[j] += offset;
  //   }
  //   for (int j = 0; j < green_size; j++) {
  //     output1.green[j] += offset;
  //   }
  //   for (int j = 0; j < blue_size; j++) {
  //     output1.blue[j] += offset;
  //   }
  // }
  return offset_and_diff;

  
}

bool MakeSpatialTemporalImages(vector<cv::Mat>& src0, vector<cv::Mat>& src1, 
                               cv::Mat& output0, cv::Mat& output1, Line& line0, Line& line1) {

  vector<cv::Mat> mats;
  int half_length = line0.half_length;
  int half_width = line0.half_width;
  double offset = line0.offset;
  Vector2d origin = line0.origin;
  double theta = line0.theta;
  for (int i = 0; i < src0.size(); i++) {
    // load the source image
    cv::Mat src = src0[i];
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols;
    cv::Mat ans(cv::Size(half_width*2+1, half_length*2+1),CV_8UC3);

    for (int row = 0; row < half_length*2+1; row++){
      for (int col = 0; col < half_width*2+1; col++) {
        cv::Vec3b pixel = ans.at<cv::Vec3b>(row, col);
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        ans.at<cv::Vec3b>(row,col) = pixel;
      }
    }

    if ((theta > PI/4 && theta < PI/4*3) || (theta > PI/4*5 && theta < PI/4*7)) {
      for (int col = origin[0] - half_length; col < origin[0] + half_length; col++) {
        int row = (offset - cos(theta) * (col-origin[0]))/sin(theta);
        row+=origin[1];
        for (int r = row - half_width; r <= row + half_width; r++) {
          if (r < nRows && r >= 0 && col < nCols && col >= 0) {
              cv::Vec3b color = src.at<cv::Vec3b>(r, col);
              ans.at<cv::Vec3b>(col-(origin[0] -half_length),r-(row-half_width)) = color;

          }
        }
      }
      
    } else {
      for (int row = origin[1] - half_length; row < origin[1] + half_length; row++) {
        int col = (offset - sin(theta) * (row-origin[1]))/cos(theta);
        col+=origin[0];
        for (int j = col - half_width; j <= col + half_width; j++) {
          if (j < nCols && j >= 0 && row < nRows && row >= 0) {
              cv::Vec3b color = src.at<cv::Vec3b>(row, j);
              ans.at<cv::Vec3b>(row-(origin[1]-half_length),j-(col-half_width)) = color;
          }
        }
      }
    }
    
    mats.push_back(ans);
  }

  int channels = mats[0].channels();
  int nRows = mats[0].rows;
  int nCols = mats[0].cols;
  //cv::Mat * ans;
  //ans = new cv::Mat(cv::Size(mats[0].cols * mats.size(),mats[0].rows),CV_8U);
  cv::Mat ret(mats[0].rows,mats[0].cols * mats.size(),CV_8UC3);

  for (int row = 0; row < mats[0].rows; row++){
    for (int col = 0; col < mats[0].cols * mats.size(); col++) {
      cv::Vec3b pixel = ret.at<cv::Vec3b>(row, col);
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
      ret.at<cv::Vec3b>(row,col) = pixel;
    }
  }


  for (int i = 0; i < mats.size(); i++) {
    cv::Mat src = mats[i];

    for (int c = 0; c < nCols; c++) {
      for (int r = 0; r < nRows; r++) {                
        cv::Vec3b color = src.at<cv::Vec3b>(r, c);
        ret.at<cv::Vec3b>(r , nCols * i + c) = color;
      }
    }
  }

  output0 = cv::Mat(ret);
  

  mats.clear();


  half_length = line1.half_length;
  half_width = line1.half_width;
  offset = line1.offset;
  origin = line1.origin;
  theta = line1.theta;

  //cout << theta << " " << offset << endl;


  for (int i = 0; i < src1.size(); i++) {
    // load the source image
    cv::Mat src = src1[i];
    int channels = src.channels();
    int nRows = src.rows;
    int nCols = src.cols;
    cv::Mat ans(cv::Size(half_width*2+1, half_length*2+1),CV_8UC3);




    for (int row = 0; row < half_length*2+1; row++){
      for (int col = 0; col < half_width*2+1; col++) {
        cv::Vec3b pixel = ans.at<cv::Vec3b>(row, col);
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
        ans.at<cv::Vec3b>(row,col) = pixel;
      }
    }

    if ((theta > PI/4 && theta < PI/4*3) || (theta > PI/4*5 && theta < PI/4*7)) {
      for (int col = origin[0] - half_length; col < origin[0] + half_length; col++) {
        int row = (offset - cos(theta) * (col-origin[0]))/sin(theta);
        row+=origin[1];
        for (int r = row - half_width; r <= row + half_width; r++) {
          if (r < nRows && r >= 0 && col < nCols && col >= 0) {
              cv::Vec3b color = src.at<cv::Vec3b>(r, col);
              ans.at<cv::Vec3b>(col-(origin[0] -half_length),r-(row-half_width)) = color;

          }
        }
      }
      
    } else {
      for (int row = origin[1] - half_length; row < origin[1] + half_length; row++) {
        int col = (offset - sin(theta) * (row-origin[1]))/cos(theta);
        col+=origin[0];
        for (int j = col - half_width; j <= col + half_width; j++) {
          if (j < nCols && j >= 0 && row < nRows && row >= 0) {
              cv::Vec3b color = src.at<cv::Vec3b>(row, j);
              ans.at<cv::Vec3b>(row-(origin[1]-half_length),j-(col-half_width)) = color;
          }
        }
      }
    }
    mats.push_back(ans);
  }


  channels = mats[0].channels();
  // cv::namedWindow( "Display window2", cv::WINDOW_AUTOSIZE );
  // cv::imshow( "Display window2", mats[0] );
  nRows = mats[0].rows;
  nCols = mats[0].cols;
  //cv::Mat * ans;
  //ans = new cv::Mat(cv::Size(mats[0].cols * mats.size(),mats[0].rows),CV_8U);
  ret = cv::Mat(mats[0].rows,mats[0].cols * mats.size(),CV_8UC3);

  for (int row = 0; row < mats[0].rows; row++){
    for (int col = 0; col < mats[0].cols * mats.size(); col++) {
      cv::Vec3b pixel = ret.at<cv::Vec3b>(row, col);
      pixel[0] = 0;
      pixel[1] = 0;
      pixel[2] = 0;
      ret.at<cv::Vec3b>(row,col) = pixel;
    }
  }


  for (int i = 0; i < mats.size(); i++) {
    cv::Mat src = mats[i];

    for (int c = 0; c < nCols; c++) {
      for (int r = 0; r < nRows; r++) {                
        cv::Vec3b color = src.at<cv::Vec3b>(r, c);
        ret.at<cv::Vec3b>(r , nCols * i + c) = color;
      }
    }
  }

  output1 = cv::Mat(ret);
  // cv::namedWindow( "Display window1", cv::WINDOW_AUTOSIZE );
  // cv::imshow( "Display window1", src1[0] );
  // cv::waitKey(0);

	return true;
}

void InitializeLine(Line& line0){
  Vector2d p1 = line0.start;
  Vector2d p2 = line0.end;
  Vector2d origin = line0.origin;
  Vector2d v1 = p1 - p2;
  
  v1.normalize();
  Vector2d v2 = p1 - origin;
  //normalize(v2);
  double dis = v2.dot(v1);
  Vector2d intersect = p1 - v1*dis;
  line0.offset = distance(origin, intersect);

  Vector2d v3 = intersect - origin;
  v3.normalize();
  Vector2d v4(1, 0);
  double product = v3.dot(v4);
  line0.theta = acos(product);
  // cout << line0.theta << endl;
  // cout << line0.offset << endl;

  line0.half_width = 1;
  line0.half_length = 250;
}

void ProcessImages(cv::Mat & mask0, cv::Mat & mask1, vector<cv::Mat>& src0, vector<cv::Mat>& src1,
                   vector<cv::Mat>& dst0, vector<cv::Mat>& dst1) {
  // Erase all pixels other than green pixels
  cv::Mat output0 = mask0.clone();
  cv::Mat output1 = mask1.clone();
  int nRows = mask0.rows;
  int nCols = mask0.cols;

  int row, col;
  for ( col = 0; col < nCols; ++col ) {
    for( row = 0; row < nRows; ++row ) {
      cv::Vec3b pixel = mask0.at<cv::Vec3b>(row, col);
      if (pixel[1] > 250 && pixel[2] < 20 && (pixel[0] < 150 && pixel[0] > 100) ) {
        //cout << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2] << endl;
        pixel[0] = 255;
        pixel[1] = 255;
        pixel[2] = 255;
      } else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output0.at<cv::Vec3b>(row, col) = pixel;

      pixel = mask1.at<cv::Vec3b>(row, col);
      if (pixel[1] > 250 && pixel[2] < 20 && (pixel[0] < 150 && pixel[0] > 100) ) {
        //cout << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2] << endl;
        pixel[0] = 255;
        pixel[1] = 255;
        pixel[2] = 255;
      } else {
        pixel[0] = 0;
        pixel[1] = 0;
        pixel[2] = 0;
      }
      output1.at<cv::Vec3b>(row, col) = pixel;
    }
  }

  int src0_size = src0.size();
  int src1_size = src1.size();


  for (int i = 0; i < src0_size; i++) {
    int channels = src0[0].channels();
    int nRows = src0[0].rows;
    int nCols = src0[0].cols;

    cv::Mat ret = src0[i].clone();

    for (int row = 0; row < nRows; row++){
      for (int col = 0; col < nCols; col++) {
        cv::Vec3b pixel = ret.at<cv::Vec3b>(row, col);
        cv::Vec3b mask_pixel = output0.at<cv::Vec3b>(row, col);
        if ((int) mask_pixel[0] == 0 && (int) mask_pixel[1] == 0 && (int) mask_pixel[2] == 0) {
          pixel[0] = 0;
          pixel[1] = 0;
          pixel[2] = 0;
        }
        ret.at<cv::Vec3b>(row,col) = pixel;
      }
    }
    dst0.push_back(ret);
  }

  for (int i = 0; i < src0_size; i++) {
    int channels = src1[0].channels();
    int nRows = src1[0].rows;
    int nCols = src1[0].cols;

    cv::Mat ret = src1[i].clone();

    for (int row = 0; row < nRows; row++){
      for (int col = 0; col < nCols; col++) {
        cv::Vec3b pixel = ret.at<cv::Vec3b>(row, col);
        cv::Vec3b mask_pixel = output1.at<cv::Vec3b>(row, col);
        //cout << (int)mask_pixel[0] << " " << (int)mask_pixel[1] << " " << (int)mask_pixel[2] << endl;
        if ((int) mask_pixel[0] == 0 && (int) mask_pixel[1] == 0 && (int) mask_pixel[2] == 0) {
          pixel[0] = 0;
          pixel[1] = 0;
          pixel[2] = 0;
        }
        ret.at<cv::Vec3b>(row,col) = pixel;
      }
    }
    dst1.push_back(ret);
  }

  cv::namedWindow( "Display window0", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window0", output0 ); 
  cv::namedWindow( "Display window1", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window1", output1 );   
  cv::waitKey(0);
}

void DrawLine(Line line, cv::Mat image) {
  int thickness = 2;
  int lineType = 8;
  int half_length = line.half_length;
  int half_width = line.half_width;
  Vector2d origin = line.origin;
  double offset = line.offset;
  double theta = line.theta;
  int startX = origin[0] + offset * cos(theta) - half_length * sin(theta);
  int startY = origin[1] + offset * sin(theta) + half_length * cos(theta);
  int endX = origin[0] + offset * cos(theta) + half_length * sin(theta);
  int endY = origin[1] + offset * sin(theta) - half_length * cos(theta);
  // cout << startX << endl;
  // cout << startY << endl;
  // cout << endX << endl;
  // cout << endY << endl;
  cv::Point start = cv::Point(startX, startY);
  cv::Point end = cv::Point(endX, endY);
  cv::line( image, start, end, cv::Scalar(255, 255, 255), thickness, lineType);
  
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " input_directory" << endl;
    return 1;
  }
  vector<cv::Mat> images0;
  vector<cv::Mat> images1;

  /* Source images paths */
  string path0 = "source/0000 copy";
  string path1 = "source/0001 copy";

  /* Initialize two lines */
  Line line0, line1;
  line0.start = Vector2d(410/FACTOR, 100/FACTOR);
  line0.end = Vector2d(600/FACTOR, 700/FACTOR);
  line0.origin = Vector2d(400/FACTOR, 300/FACTOR);

  line1.start = Vector2d(410/FACTOR, 100/FACTOR);
  line1.end = Vector2d(550/FACTOR, 700/FACTOR);
  line1.origin = Vector2d(400/FACTOR, 300/FACTOR);

  InitializeLine(line0);
  InitializeLine(line1);
  
  vector<Line> lines0;
  vector<Line> lines1;

  /* Initialize all lines */
  for (int i = -10; i < 10; i++) {
    for (int j = -10; j < 10; j++) {
      double offset = line0.offset + i * 5.0 / FACTOR;
      double theta = line0.theta + j * 0.02;
      Line newLine;
      newLine.offset = offset;
      newLine.theta = theta;
      newLine.half_width = 1;
      newLine.half_length = 250/FACTOR;
      newLine.origin = Vector2d(400/FACTOR, 300/FACTOR);
      lines0.push_back(newLine);
    }
  }

  for (int i = -10; i < 10; i++) {
    for (int j = -10; j < 10; j++) {
      double offset = line1.offset + i * 1.0;
      double theta = line1.theta + j * 0.02;
      Line newLine;
      newLine.offset = offset;
      newLine.theta = theta;
      newLine.half_width = 1;
      newLine.half_length = 250/FACTOR;
      newLine.origin = Vector2d(400/FACTOR, 300/FACTOR);
      lines1.push_back(newLine);
    }
  }

  ReadImages(path0, path1, images0, images1);

  int best_i = -1;
  int best_j = -1;
  int best_score = -99999999;

  /* Find best lines */

  // for (int i = 0; i < lines0.size(); i++) {
  //   cout << i/(double) lines0.size() * 100 << " percent" << endl;
  //   for (int j = 0; j < lines1.size(); j++) {
  //     Columns output0;
  //     Columns output1;

  //     pair<int, int> ans = MakeColumns(images0, images1, output0, output1, lines0[i], lines1[j]);
      
  //     if (ans.second > best_score) {
        
  //       best_score = ans.second;
  //       best_i = i;
  //       best_j = j;
  //       cout << best_score << " " << best_i << " " << best_j << endl;
        
  //     }

  //   }
  // }
  // cout << "Best i = " << best_i << "Best j = " << best_j << endl;


  int i = 160; // best i
  int j = 304; // best j

  line0 = lines0[i];
  line1 = lines1[j];

  cv::Mat spatial0;
  cv::Mat spatial1;

  /* Make spatial-temporal images*/
  // MakeSpatialTemporalImages(images0, images1, spatial0, spatial1, line0, line1);

  // cv::namedWindow( "Display window0", cv::WINDOW_AUTOSIZE );
  // cv::imshow( "Display window0", spatial0 ); 
  // cv::namedWindow( "Display window1", cv::WINDOW_AUTOSIZE );
  // cv::imshow( "Display window1", spatial1 );   
  // cv::waitKey(0);


  /* Draw best line */ 
  // DrawLine(lines0[i], images0[0]);
  // DrawLine(lines1[j], images1[0]);

  /* Draw all lines */
  for (int i = 0; i < lines0.size(); i++) {
    DrawLine(lines0[i], images0[0]);
  }
  for (int j = 0; j < lines1.size(); j++) {
    DrawLine(lines1[j], images1[0]);
  }
  
  
  

  cv::namedWindow( "Display window0", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window0", images0[0] );   

  cv::namedWindow( "Display window1", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window1", images1[0] );   
  cv::waitKey(0);
  
  
}
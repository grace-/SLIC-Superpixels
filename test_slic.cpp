/*
 * test_slic.cpp.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp.
 */
 
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;

#include "slic.h"

// ./slic dog.png 400 40 test.png

int main(int argc, char *argv[]) {
  /* Load the image and convert to Lab colour space. */
  cv::Mat image = cv::imread(argv[1], 1);
  cv::Mat lab_image;
  cv::cvtColor(image, lab_image, CV_BGR2Lab);
  
  /* Yield the number of superpixels and weight-factors from the user. */
  int w = image.cols, h = image.rows;
  int nr_superpixels = atoi(argv[2]);
  int nc = atoi(argv[3]);
  
  double step = sqrt((w * h) / (double) nr_superpixels);
  
  /* Perform the SLIC superpixel algorithm. */
  Slic slic;
  slic.generate_superpixels(lab_image, step, nc);
  slic.create_connectivity(lab_image);
  
  std::vector<std::vector<cv::Point> > pointSets = slic.generatePointSets(image);
  
  
  /* Display the contours and show the result. */
  //  slic.colour_with_cluster_means(image);
  slic.display_contours(image, cv::Vec3b(0,0,255));
 
  std::vector<std::vector<int> > edges = slic.generateGraph(image);
  slic.displayGraph(image, edges, cv::Vec3b(0,255,255));

  slic.display_center_grid(image, cv::Vec3b(0, 0, 0));
  
  //  std::vector<std::vector<cv::Point> > polys = slic.generateBoundingPolys(image);
  //  slic.displayBoundingPolys(image, polys, cv::Vec3b(255,0,0));
  //  std::vector<cv::Rect> boxes = slic.generateBoundingBoxes(image);
  //  slic.displayBoundingBoxes(image, boxes, cv::Vec3b(0, 255, 255));
  //  std::vector<cv::RotatedRect> boxes = slic.generateRotBoundingBoxes(image);
  //  slic.displayRotBoundingBoxes(image, boxes, cv::Vec3b(0, 255, 255));
  //  std::cout << boxes[0].size() << std::endl;

  cv::namedWindow("result", CV_WINDOW_AUTOSIZE);
  cv::imshow("result", image);
  cv::waitKey(0);
  cv::imwrite(argv[4], image);
}

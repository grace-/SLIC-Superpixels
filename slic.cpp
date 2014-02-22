#include "slic.h"

/*
 * Constructor. Nothing is done here.
 */
Slic::Slic() {

}

/*
 * Destructor. Clear any present data.
 */
Slic::~Slic() {
  clear_data();
}

/*
 * Clear the data as saved by the algorithm.
 *
 * Input : -
 * Output: -
 */
void Slic::clear_data() {
  clusters.clear();
  distances.clear();
  centers.clear();
  center_counts.clear();
}
 
/*
 * Initialize the cluster centers and initial values of the pixel-wise cluster
 * assignment and distance values.
 *
 * Input : The image (IplImage*).
 * Output: -
 */
void Slic::init_data(const cv::Mat &image) {
  /* Initialize the cluster and distance matrices. */
  for (int i = 0; i < image.cols; i++) { 
    vector<int> cr;
    vector<double> dr;
    for (int j = 0; j < image.rows; j++) {
      cr.push_back(-1);
      dr.push_back(FLT_MAX);
    }
    clusters.push_back(cr);
    distances.push_back(dr);
  }
  
  /* Initialize the centers and counters. */
  for (int i = step; i < image.cols - step/2; i += step) {
    for (int j = step; j < image.rows - step/2; j += step) {
      vector<double> center;
      /* Find the local minimum (gradient-wise). */
      cv::Point nc = find_local_minimum(image, cv::Point(i,j));
      
      /* Generate the center vector. */
      center.push_back(image.data[nc.y*image.step+nc.x*image.channels()+0]);
      center.push_back(image.data[nc.y*image.step+nc.x*image.channels()+1]);
      center.push_back(image.data[nc.y*image.step+nc.x*image.channels()+2]);
      center.push_back(nc.x);
      center.push_back(nc.y);
      
      /* Append to vector of centers. */
      centers.push_back(center);
      center_counts.push_back(0);
    }
  }
}

/*
 * Compute the distance between a cluster center and an individual pixel.
 *
 * Input : The cluster index (int), the pixel (cv::Point), and the Lab values of
 *         the pixel (cv::Scalar).
 * Output: The distance (double).
 */
double Slic::compute_dist(int ci, cv::Point pixel, unsigned char x0, unsigned char x1, unsigned char x2) {
  double dc = sqrt(pow(centers[ci][0] - x0, 2) + pow(centers[ci][1] - x1, 2) + pow(centers[ci][2] - x2, 2));
  double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));    
  return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));
}

/*
 * Find a local gradient minimum of a pixel in a 3x3 neighbourhood. This
 * method is called upon initialization of the cluster centers.
 *
 * Input : The image (IplImage*) and the pixel center (cv::Point).
 * Output: The local gradient minimum (cv::Point).
 */
cv::Point Slic::find_local_minimum(const cv::Mat &image, cv::Point center) {
  double min_grad = FLT_MAX;
  cv::Point loc_min = cv::Point(center.x, center.y);
  
  for (int i = center.x-1; i < center.x+2; i++) {
    for (int j = center.y-1; j < center.y+2; j++) {
      /* Convert colour values to grayscale values. */
      double i1 = image.data[(j+1)*image.step+(i  )*image.channels()+0];
      double i2 = image.data[(j  )*image.step+(i+1)*image.channels()+0];
      double i3 = image.data[(j  )*image.step+(i  )*image.channels()+0];
      
      /* Compute horizontal and vertical gradients and keep track of the minimum. */
      if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
	min_grad = fabs(i1 - i3) + fabs(i2 - i3);
	loc_min.x = i;
	loc_min.y = j;
      }
    }
  }    
  return loc_min;
}

/*
 * Compute the over-segmentation based on the step-size and relative weighting
 * of the pixel and colour values.
 *
 * Input : The Lab image (IplImage*), the stepsize (int), and the weight (int).
 * Output: -
 */
void Slic::generate_superpixels(const cv::Mat &image, int step, int nc) {
  this->step = step;
  this->nc = nc;
  this->ns = step;
  
  /* Clear previous data (if any), and re-initialize it. */
  clear_data();
  init_data(image);
  
  /* Run EM for 10 iterations (as prescribed by the algorithm). */
  for (int i = 0; i < NR_ITERATIONS; i++) {
    /* Reset distance values. */
    for (int j = 0; j < image.cols; j++) {
      for (int k = 0;k < image.rows; k++) {
	distances[j][k] = FLT_MAX;
      }
    }
    
    for (int j = 0; j < (int) centers.size(); j++) {
      /* Only compare to pixels in a 2 x step by 2 x step region. */
      for (int k = centers[j][3] - step; k < centers[j][3] + step; k++) {
	for (int l = centers[j][4] - step; l < centers[j][4] + step; l++) {	  
	  if (k >= 0 && k < image.cols && l >= 0 && l < image.rows) {
	    double d = compute_dist(j, cv::Point(k,l), image.data[l*image.step+k*image.channels()+0],
				    image.data[l*image.step+k*image.channels()+1],
				    image.data[l*image.step+k*image.channels()+2]);
	    
	    /* Update cluster allocation if the cluster minimizes the
	       distance. */
	    if (d < distances[k][l]) {
	      distances[k][l] = d;
	      clusters[k][l] = j;
	    }
	  }
	}
      }
    }
    
    /* Clear the center values. */
    for (int j = 0; j < (int) centers.size(); j++) {
      centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
      center_counts[j] = 0;
    }
    
    /* Compute the new cluster centers. */
    for (int j = 0; j < image.cols; j++) {
      for (int k = 0; k < image.rows; k++) {
	int c_id = clusters[j][k];
        
	if (c_id != -1) {
	  centers[c_id][0] += image.data[k*image.step+j*image.channels()+0];
	  centers[c_id][1] += image.data[k*image.step+j*image.channels()+1];
	  centers[c_id][2] += image.data[k*image.step+j*image.channels()+2];
	  centers[c_id][3] += j;
	  centers[c_id][4] += k;
          
	  center_counts[c_id] += 1;
	}
      }
    }
    
    /* Normalize the clusters. */
    for (int j = 0; j < (int) centers.size(); j++) {
      centers[j][0] /= center_counts[j];
      centers[j][1] /= center_counts[j];
      centers[j][2] /= center_counts[j];
      centers[j][3] /= center_counts[j];
      centers[j][4] /= center_counts[j];
    }
  }
}
void Slic::generate_superpixels2(const cv::Mat &image, int nr_superpixels, int nc) {
  int w = image.cols, h = image.rows;
  double step = sqrt((w * h) / (double) nr_superpixels);
  generate_superpixels(image, round(step), nc);
}

std::vector<std::vector<int> > Slic::generateGraph(const cv::Mat &im) {
  std::vector<std::vector<int> > edges;
  {
    std::vector<int> row(centers.size(),0);
    for ( int i = 0; i < centers.size(); ++i ) edges.push_back(row);
  }  
  if ( centers.size() ) {
    for (int i = 1; i < im.cols; ++i) {
      for (int j = 1; j < im.rows; ++j) {
	int w = clusters[i  ][j  ];
	int u = clusters[i-1][j  ];
	int v = clusters[i  ][j-1];
	if ( !(w==u) ) { edges[w][u] = 1; edges[u][w] = 1; }
	if ( !(w==v) ) { edges[w][v] = 1; edges[v][w] = 1; }
      }
    }
  }  
  return edges;
}
void Slic::displayGraph(cv::Mat &im, std::vector<std::vector<int> > &edges, cv::Vec3b colour) {
  for ( int i = 0; i < edges.size(); ++i ) {
    for ( int j = i; j < edges.size(); ++j ) {
      if ( edges[i][j] )
	cv::line(im, cv::Point(centers[i][3], centers[i][4]),
		 cv::Point(centers[j][3], centers[j][4]),
		 (cv::Scalar)colour, 1);
    }
  }
}

/*
 * Enforce connectivity of the superpixels. This part is not actively discussed
 * in the paper, but forms an active part of the implementation of the authors
 * of the paper.
 *
 * Input : The image (IplImage*).
 * Output: -
 */
void Slic::create_connectivity(const cv::Mat &image) {
  int label = 0, adjlabel = 0;

  if ( centers.size() ) {
    const int lims = (image.cols * image.rows) / ((int)centers.size());
    
    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};
    
    /* Initialize the new cluster matrix. */
    vec2di new_clusters;
    for (int i = 0; i < image.cols; i++) { 
      vector<int> nc;
      for (int j = 0; j < image.rows; j++) {
	nc.push_back(-1);
      }
      new_clusters.push_back(nc);
    }
    
    for (int i = 0; i < image.cols; i++) {
      for (int j = 0; j < image.rows; j++) {
	if (new_clusters[i][j] == -1) {
	  vector<cv::Point> elements;
	  elements.push_back(cv::Point(i, j));
	  
	  /* Find an adjacent label, for possible use later. */
	  for (int k = 0; k < 4; k++) {
	    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
	    
	    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
	      if (new_clusters[x][y] >= 0) {
		adjlabel = new_clusters[x][y];
	      }
	    }
	  }
	  
	  int count = 1;
	  for (int c = 0; c < count; c++) {
	    for (int k = 0; k < 4; k++) {
	      int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
	      
	      if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
		if (new_clusters[x][y] == -1 && clusters[i][j] == clusters[x][y]) {
		  elements.push_back(cv::Point(x, y));
		  new_clusters[x][y] = label;
		  count += 1;
		}
	      }
	    }
	  }
	  
	  /* Use the earlier found adjacent label if a segment size is
	     smaller than a limit. */
	  if (count <= lims >> 2) {
	    for (int c = 0; c < count; c++) {
	      new_clusters[elements[c].x][elements[c].y] = adjlabel;
	    }
	    label -= 1;
	  }
	  label += 1;
	}
      }
    }
  } else {
    std::cerr << "no pixel centers found!\n"; 
  }
}

/*
 * Display the cluster centers.
 *
 * Input : The image to display upon (IplImage*) and the colour (cv::Scalar).
 * Output: -
 */
void Slic::display_center_grid(cv::Mat &image, cv::Vec3b colour) {
  for (int i = 0; i < (int) centers.size(); i++) {
    cv::circle(image, cv::Point(centers[i][3], centers[i][4]), 1, (cv::Scalar)colour, 1);
  }
}

/*
 * Display a single pixel wide contour around the clusters.
 *
 * Input : The target image (IplImage*) and contour colour (cv::Scalar).
 * Output: -
 */
void Slic::display_contours(cv::Mat &image, cv::Vec3b colour) {
  const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
  const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
  
  /* Initialize the contour vector and the matrix detailing whether a pixel
   * is already taken to be a contour. */
  vector<cv::Point> contours;
  vec2db istaken;
  for (int i = 0; i < image.cols; i++) { 
    vector<bool> nb;
    for (int j = 0; j < image.rows; j++) {
      nb.push_back(false);
    }
    istaken.push_back(nb);
  }
  
  /* Go through all the pixels. */
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      int nr_p = 0;
      
      /* Compare the pixel to its 8 neighbours. */
      for (int k = 0; k < 8; k++) {
	int x = i + dx8[k], y = j + dy8[k];
        
	if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
	  if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
	    nr_p += 1;
	  }
	}
      }
      
      /* Add the pixel to the contour list if desired. */
      if (nr_p >= 2) {
	contours.push_back(cv::Point(i,j));
	istaken[i][j] = true;
      }
    }
  }
  
  /* Draw the contour pixels. */
  for (int i = 0; i < (int)contours.size(); i++) {
    image.data[contours[i].y*image.step+contours[i].x*image.channels()+0] = colour.val[0];
    image.data[contours[i].y*image.step+contours[i].x*image.channels()+1] = colour.val[1];
    image.data[contours[i].y*image.step+contours[i].x*image.channels()+2] = colour.val[2];
  }
}

/*
 * Give the pixels of each cluster the same colour values. The specified colour
 * is the mean RGB colour per cluster.
 *
 * Input : The target image (IplImage*).
 * Output: -
 */
void Slic::colour_with_cluster_means(cv::Mat &image) {
  vector<cv::Vec3f> colours(centers.size());
  
  /* Gather the colour values per cluster. */
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      int index = clusters[i][j];
      colours[index].val[0] += static_cast<float>(image.data[j*image.step+i*image.channels()+0]);
      colours[index].val[1] += static_cast<float>(image.data[j*image.step+i*image.channels()+1]);
      colours[index].val[2] += static_cast<float>(image.data[j*image.step+i*image.channels()+2]); 
    }
  }
  
  /* Divide by the number of pixels per cluster to get the mean colour. */
  for (int i = 0; i < (int)colours.size(); i++) {
    colours[i].val[0] /= static_cast<float>(center_counts[i]);
    colours[i].val[1] /= static_cast<float>(center_counts[i]);
    colours[i].val[2] /= static_cast<float>(center_counts[i]);
  }
  
  /* Fill in. */
  for (int i = 0; i < image.cols; i++) {
    for (int j = 0; j < image.rows; j++) {
      cv::Vec3f ncolour = colours[clusters[i][j]];
      image.data[j*image.step+i*image.channels()+0] = static_cast<unsigned char>(round(ncolour.val[0]));
      image.data[j*image.step+i*image.channels()+1] = static_cast<unsigned char>(round(ncolour.val[1]));
      image.data[j*image.step+i*image.channels()+2] = static_cast<unsigned char>(round(ncolour.val[2]));
    }
  }
}

std::vector<std::vector<cv::Point> > Slic::generatePointSets(const cv::Mat &image) {
  std::vector<std::vector<cv::Point> > pointSets(centers.size());
  for (int i = 0; i < image.cols; ++i) {
    for (int j = 0; j < image.rows; ++j) {
      int idx = clusters[i][j];
      pointSets[idx].push_back(cv::Point(i,j));
    }
  }
  return pointSets;
}

std::vector<cv::Rect> Slic::generateBoundingBoxes(const cv::Mat &image) {
  std::vector<cv::Rect> R;
  std::vector<std::vector<cv::Point> > pointSets = generatePointSets(image);
  for ( int i = 0; i < pointSets.size(); ++i ) {
    cv::Rect r = cv::boundingRect(cv::Mat(pointSets[i]));
    R.push_back(r);
  }
  return R;
}
void Slic::displayBoundingBoxes(cv::Mat &image, std::vector<cv::Rect> &boxes, cv::Vec3b colour) {
  for ( int i = 0; i < boxes.size(); ++i ) {
    cv::Rect &box = boxes[i];
    cv::rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), (cv::Scalar)colour, 1);
  }
}

std::vector<cv::RotatedRect> Slic::generateRotBoundingBoxes(const cv::Mat &image) {
  std::vector<cv::RotatedRect> R;
  std::vector<std::vector<cv::Point> > pointSets = generatePointSets(image);
  for ( int i = 0; i < pointSets.size(); ++i ) {
    cv::RotatedRect r = cv::minAreaRect(cv::Mat(pointSets[i]));
    R.push_back(r);
  }
  return R;
}

void Slic::displayRotBoundingBoxes( cv::Mat &image, std::vector<cv::RotatedRect> &boxes, cv::Vec3b colour ) {
  for ( int i = 0; i < boxes.size(); ++i ) {
    cv::Point2f vertices[4];
    boxes[i].points(vertices);
    for ( int j = 0; j < 4; ++j ) {
      cv::line(image, vertices[j], vertices[(j+1)%4], (cv::Scalar)colour, 1);
    }    
  }
}

std::vector<std::vector<cv::Point> > Slic::generateBoundingPolys( const cv::Mat &image ) {
  std::vector<std::vector<cv::Point> > R;
  std::vector<std::vector<cv::Point> > pointSets = generatePointSets(image);
  for ( int i = 0; i < pointSets.size(); ++i ) {   
    std::vector<cv::Point> &pointSet = pointSets[i];
    std::vector<int> hull;
    cv::convexHull(cv::Mat(pointSet), hull, CV_CLOCKWISE);
    std::vector<cv::Point> R_;
    for ( int j = 0; j < hull.size(); ++j ) { R_.push_back(pointSet[hull[j]]); }
    R.push_back(R_);
  }  
  return R;
}

void Slic::displayBoundingPolys( cv::Mat &image, std::vector<std::vector<cv::Point> > &hulls, cv::Vec3b colour) {
  for ( int i = 0; i < hulls.size(); ++i ) {
    std::vector<cv::Point> &hull = hulls[i];
    cv::Point pt0 = hull[hull.size()-1];
    for ( int j = 0; j < hull.size(); ++j ) {
      cv::Point &pt = hull[j];
      cv::line(image, pt0, pt, (cv::Scalar)colour, 1, CV_AA);
      pt0 = pt;
    }
  }
}




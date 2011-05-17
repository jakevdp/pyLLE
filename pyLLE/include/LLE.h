#ifndef LLE_H
#define LLE_H

#include "MatVec.h"

typedef Matrix<double,DENSE> Mat_data_t;
typedef Vector<double,GEN>   Vec_data_t;
typedef Matrix<int,DENSE>    Mat_index_t;
typedef Vector<int,GEN>      Vec_index_t;

/************************************************************
 * compute_neighbors :
 *   inputs:  training_data : rank [D x N] matrix: N points
 *                             in D dimensions
 *   outputs: neighbors     : rank [k x N] matrix: on output,
 *                             n[i,j] holds the index of the i^th
 *                             nearest neighbor of point j
 ************************************************************/
void compute_neighbors(const Mat_data_t& training_data, 
		       Mat_index_t& neighbors,
		       bool use_tree = false,
		       int verbose = 0);


/************************************************************
 * compute_neighbors :
 *   inputs:  training_data : rank [D x N] matrix: N points
 *                             in D dimensions
 *            test_data     : rank [D x N2] matrix: N2 points
 *                             in D dimensions
 *   outputs: neighbors     : rank [k x N2] matrix: on output,
 *                             n[i,j] holds the index of the i^th
 *                             nearest neighbor (in training_data)
 *                             of point j (in test_data)
 ************************************************************/
void compute_neighbors(const Mat_data_t& training_data, 
		       const Mat_data_t& test_data,
		       Mat_index_t& neighbors,
		       bool use_tree = false,
		       int verbose = 0);

/************************************************************
 * compute_sigma :
 *  inputs:  training_data : rank [D x N] matrix: N points
 *                             in D dimensions
 *           neighbors     : rank [k x N] matrix
 *
 *  outputs: sigma         : rank-N array
 *
 ************************************************************/
void compute_sigma(const Mat_data_t& training_data,
		   int k,
		   int d_out,
		   Vec_data_t& sigma,
		   int verbose = 0);

void compute_sigma(const Mat_data_t& training_data,
		   const Mat_index_t& neighbors,
		   int d_out,
		   Vec_data_t& sigma,
		   int verbose = 0);

/************************************************************
 * compute_LLE_weights :
 *  
 ************************************************************/
void compute_LLE_weights(const Mat_data_t& training_data,
			 int k,
			 Mat_data_t& weight_matrix,
			 int verbose = 0);

void compute_LLE_weights(const Mat_data_t& training_data,
			 const Mat_index_t& neighbors,
			 Mat_data_t& weight_matrix,
			 int verbose = 0);

/************************************************************
 * compute_LLE_weights :
 *  
 ************************************************************/
void compute_LLE_weights(const Mat_data_t& training_data,
			 const Mat_data_t& test_data,
			 int k,
			 Mat_data_t& weight_matrix,
			 int verbose = 0);

void compute_LLE_weights(const Mat_data_t& training_data,
			 const Mat_data_t& test_data,
			 const Mat_index_t& neighbors,
			 Mat_data_t& weight_matrix,
			 int verbose = 0);

/************************************************************
 * compute_HLLE_weights :
 *  
 ************************************************************/
void compute_HLLE_weights(const Mat_data_t& training_data,
			  int k,
			  Mat_data_t& weight_matrix,
			  int d_out,
			  int verbose = 0);

void compute_HLLE_weights(const Mat_data_t& training_data,
			  const Mat_index_t& neighbors,
			  Mat_data_t& weight_matrix,
			  int verbose = 0);

/************************************************************
 * compute_LLE_d_out :
 *
 ************************************************************/
int compute_LLE_dim(const Mat_data_t& training_data,
		    int k,
		    double var,
		    int verbose = 0);

int compute_LLE_dim(const Mat_data_t& training_data,
		    const Mat_index_t& neighbors,
		    double var,
		    int verbose = 0);

/************************************************************
 * LLE
 *
 ************************************************************/
void LLE(const Mat_data_t& training_data,
	 int k,
	 Mat_data_t& projection,
	 int verbose = 0);

void LLE(const Mat_data_t& training_data,
	 const Mat_index_t& neighbors,
	 Mat_data_t& projection,
	 int verbose = 0);


/************************************************************
 * HLLE
 *
 ************************************************************/
void HLLE(const Mat_data_t& training_data,
	  int k,
	  Mat_data_t& projection,
	  int verbose = 0);

void HLLE(const Mat_data_t& training_data,
	  const Mat_index_t& neighbors,
	  Mat_data_t& projection,
	  int verbose = 0);


/************************************************************
 * MLLE
 *
 ************************************************************/
void MLLE(const Mat_data_t& training_data,
	  int k,
	  Mat_data_t& projection,
	  double TOL = 1E-12,
	  int verbose = 0);

void MLLE(const Mat_data_t& training_data,
	  const Mat_index_t& neighbors,
	  Mat_data_t& projection,
	  double TOL = 1E-12,
	  int verbose = 0);

/************************************************************
 * project_onto_LLE
 *
 ************************************************************/
void project_onto_LLE(const Mat_data_t& training_data,
		      const Mat_data_t& test_data,
		      int k,
		      Mat_data_t& test_proj,
		      int verbose = 0);

void project_onto_LLE(const Mat_data_t& training_data,
		      const Mat_data_t& training_proj,
		      const Mat_data_t& test_data,
		      int k,
		      Mat_data_t& test_proj,
		      int verbose = 0);

void project_onto_LLE(const Mat_data_t& training_data,
		      const Mat_data_t& training_proj,
		      const Mat_data_t& test_data,
		      const Mat_index_t& test_neighbors,
		      Mat_data_t& test_proj,
		      int verbose = 0);
/************************************************************
 * RLLE routines
 *
 ************************************************************/
void compute_RLLE_scores(const Mat_data_t& training_data,
			 int k,
			 Vec_data_t& r_scores,
			 int d_out,
			 int verbose = 0);

void compute_RLLE_scores(const Mat_data_t& training_data,
			 const Mat_index_t& neighbors,
			 Vec_data_t& r_scores,
			 int d_out,
			 int verbose = 0);

void compute_weighted_neighbors(const Mat_data_t& training_data, 
				const Vec_data_t& r_scores,
				int r,
				Mat_index_t& neighbors,
				int verbose = 0);
			 
void RLLE1(const Mat_data_t& training_data,
	   int k,	
	   Mat_data_t& projection,
	   double r,
	   char proj_type = 'L',
	   int verbose = 0);

void RLLE1(const Mat_data_t& training_data,
	   const Mat_index_t& neighbors,	
	   Mat_data_t& projection,
	   double r,
	   char proj_type = 'L',
	   int verbose = 0);
			 
void RLLE2(const Mat_data_t& training_data,
	   int k,	
	   Mat_data_t& projection,
	   int r,
	   int verbose = 0);
			 
void RLLE2(const Mat_data_t& training_data,
	   const Mat_index_t& neighbors,	
	   Mat_data_t& projection,
	   int r,
	   int verbose = 0);
	       
void compute_mean_dist(const Mat_data_t& training_data,
		       Vec_data_t& mean_distances,
		       int verbose = 0);
		       

#endif // LLE_H

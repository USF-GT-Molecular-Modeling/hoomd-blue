// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "PCNDAngleForceGPU.cuh"

#include "hoomd/TextureTools.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

using namespace hoomd;

#include <assert.h>

// small number. cutoff for ignoring the angle as being ill defined.
#define SMALL Scalar(0.001)

/*! \file PCNDAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the PCND angle forces. Used by
    PCNDAngleForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for caculating PCND angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params Parameters for the PCND force
    \param seed User chosen random number seed
*/
__global__ void gpu_compute_PCND_angle_forces_kernel(Scalar4* d_force,
                                                     Scalar* d_virial,
						     const unsigned int* d_tag,
                                                     const size_t virial_pitch,
                                                     const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     BoxDim box,
                                                     const group_storage<3>* alist,
                                                     const unsigned int* apos_list,
                                                     const unsigned int pitch,
                                                     const unsigned int* n_angles_list,
                                                     Scalar2* d_params,
                                                     uint64_t timestep)
                                                     //uint64_t PCNDtimestep)
						     //uint16_t seed)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N)
        return;
	
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c-triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx]; // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos, c_pos; // allocate space for the a, b, and c atom in the a-b-c triplet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
     
    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {		
        group_storage<3> cur_angle = alist[pitch * angle_idx + idx];
        int cur_angle_x_idx = cur_angle.idx[0];
	int cur_angle_y_idx = cur_angle.idx[1];
	int cur_angle_type = cur_angle.idx[2];

	int cur_angle_abc = apos_list[pitch * angle_idx + idx];

	// get the a-particle's position (MEM TRANSFER: 16 bytes)
	Scalar4 x_postype = d_pos[cur_angle_x_idx];
	Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
	// get the c-particle's position (MEM TRANSFER:16 bytes)
	Scalar4 y_postype = d_pos[cur_angle_y_idx];
	Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

	if (cur_angle_abc == 1)
	    {
	    a_pos = x_pos;
	    c_pos = y_pos;

	    // calculate dr for a-b, c-b, and a-c
	    Scalar3 dac = a_pos - c_pos;

	    // apply periodic boundary conditions
	    dac = box.minImage(dac);

            // get the angle parameters (MEM TRANSFER: 8 bytes)
	    Scalar2 params = __ldg(d_params + cur_angle_type);
	    Scalar Xi = params.x;
	    Scalar Tau = params.y;
	
	    Scalar rsqac = dot(dac, dac);
	    Scalar rac = sqrtf(rsqac);
	    dac = dac / rac;

	    // read in the tag of our particle.
	    unsigned int ptag = d_tag[idx];

	    uint16_t seed = 1;

	    // Initialize the Random Number Generator and generate the 6 random numbers
	    RandomGenerator rng(hoomd::Seed(RNGIdentifier::PCNDAngleForceCompute, timestep, seed),
			        hoomd::Counter(ptag));
	    UniformDistribution<Scalar> uniform(Scalar(0), Scalar(1));

	    Scalar a = uniform(rng);
	    Scalar b = uniform(rng);

            Scalar E = exp(-1 / Tau);
            Scalar mag = d_force[idx].w;

	    if (Xi != 0)
                {
                Scalar h = 0;
	        h = Xi * sqrtf(-2 * (1 - E * E) * logf(a)) * cosf(2 * 3.1415926535897 * b);
	        mag = mag * E + h;
	        if (timestep == 0)
                    {
                    mag = Xi * sqrtf(-2 * logf(a)) * cosf(2 * 3.1415926535897 * b);
	            }
                Scalar mag_bound = 0;
	        mag_bound = Xi * sqrtf(-2 * logf(0.001));
                if (mag > mag_bound)
                    {
		    mag = mag_bound;
		    }
	        else if (mag < -mag_bound)
                    {
                    mag = -mag_bound;
                    } 

	        force_idx.x += dac.x * mag;
	        force_idx.y += dac.y * mag;
	        force_idx.z += dac.z * mag;
	        force_idx.w += mag;
	        }
	    else
                {
                force_idx.x += 0;
	        force_idx.y += 0;
	        force_idx.z += 0;
	        force_idx.w += 0;
	        }
            }
        
	if (cur_angle_abc == 0 && n_angles == 2)
            {
            force_idx.x += 0;
	    force_idx.y += 0;
	    force_idx.z += 0;
	    force_idx.w += 0;
	    }

	if (cur_angle_abc == 2 && n_angles == 2)
            {
            force_idx.x += 0;
	    force_idx.y += 0;
	    force_idx.z += 0;
	    force_idx.w += 0;
	    }
	}
	// Now that the force calculation is complete, write out the result
        d_force[idx] = force_idx;
        }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params Xi and Tau params packed as Scalar2 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param seed User chosen random number seed

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains Xi
    the RMS force magnitude and the y component contains Tau the correlation time.
*/
hipError_t gpu_compute_PCND_angle_forces(Scalar4* d_force,
                                         Scalar* d_virial,
                                         const size_t virial_pitch,
					 const unsigned int* d_tag,
                                         const unsigned int N,
                                         const Scalar4* d_pos,
                                         const BoxDim& box,
                                         const group_storage<3>* atable,
                                         const unsigned int* apos_list,
                                         const unsigned int pitch,
                                         const unsigned int* n_angles_list,
                                         Scalar2* d_params,
                                         unsigned int n_angle_types,
                                         int block_size,
                                         uint64_t timestep)
                                         //uint64_t PCNDtimestep)
					 //uint16_t seed)
    {
    assert(d_params);
    
    static unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_PCND_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    
    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_PCND_angle_forces_kernel),
		        dim3(grid),
			dim3(threads),
			0,
			0,
			d_force,
                        d_virial,
			d_tag,
                        virial_pitch,
                        N,
                        d_pos,
                        box,
                        atable,
                        apos_list,
                        pitch,
                        n_angles_list,
                        d_params,
                        timestep);
                        //PCNDtimestep);
			//seed);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

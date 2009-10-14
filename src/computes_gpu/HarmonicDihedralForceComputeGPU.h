/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: akohlmey

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include "HarmonicDihedralForceCompute.h"
#include "HarmonicDihedralForceGPU.cuh"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

/*! \file HarmonicDihedralForceComputeGPU.h
    \brief Declares the HarmonicDihedralForceGPU class
*/

#ifndef __HARMONICDIHEDRALFORCECOMPUTEGPU_H__
#define __HARMONICDIHEDRALFORCECOMPUTEGPU_H__

//! Implements the harmonic dihedral force calculation on the GPU
/*! HarmonicDihedralForceComputeGPU implements the same calculations as HarmonicDihedralForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as float2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in dihedralforce_kernel.cu.

    \ingroup computes
*/
class HarmonicDihedralForceComputeGPU : public HarmonicDihedralForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicDihedralForceComputeGPU(boost::shared_ptr<SystemDefinition> system);
        //! Destructor
        ~HarmonicDihedralForceComputeGPU();
        
        //! Sets the block size to run on the device
        /*! \param block_size Block size to set
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }
            
        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, int sign, unsigned int multiplicity);
        
    protected:
        int m_block_size;               //!< Block size to run calculation on
        vector<float4 *> m_gpu_params;  //!< Parameters stored on the GPU (k,sign,m)
        float4 *m_host_params;          //!< Host parameters -- padded to float4
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the DihedralForceComputeGPU class to python
void export_HarmonicDihedralForceComputeGPU();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif


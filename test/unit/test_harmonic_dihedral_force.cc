/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "HarmonicDihedralForceCompute.h"
#include "ConstForceCompute.h"
#ifdef ENABLE_CUDA
#include "HarmonicDihedralForceComputeGPU.h"
#endif

#include <stdio.h>

#include "Initializers.h"

using namespace std;
using namespace boost;

//! Name the boost unit test module
#define BOOST_TEST_MODULE DihedralForceTests
#include "boost_utf_configure.h"

//! Typedef to make using the boost::function factory easier
typedef boost::function<shared_ptr<HarmonicDihedralForceCompute>  (shared_ptr<SystemDefinition> sysdef)> dihedralforce_creator;

//! Perform some simple functionality tests of any BondForceCompute
void dihedral_force_basic_tests(dihedralforce_creator tf_creator, boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    /////////////////////////////////////////////////////////
    // start with the simplest possible test: 4 particles in a huge box with only one dihedral type !!!! NO DIHEDRALS
    shared_ptr<SystemDefinition> sysdef_4(new SystemDefinition(4, BoxDim(1000.0), 1, 0, 0, 1, 0, exec_conf));
    shared_ptr<ParticleData> pdata_4 = sysdef_4->getParticleData();

    pdata_4->setPosition(0,make_scalar3(10.0,1.0,2.0));
    pdata_4->setPosition(1,make_scalar3(1.0,1.0,1.0));
    pdata_4->setPosition(2,make_scalar3(6.0,-7.0,8.0));
    pdata_4->setPosition(3,make_scalar3(9.0,50.0,11.0));
    
    /*
        printf(" Particle 1: x = %f  y = %f  z = %f \n", h_pos.data[0].x, h_pos.data[0].y, h_pos.data[0].z);
        printf(" Particle 2: x = %f  y = %f  z = %f \n", h_pos.data[1].x, h_pos.data[1].y, h_pos.data[1].z);
        printf(" Particle 3: x = %f  y = %f  z = %f \n", h_pos.data[2].x, h_pos.data[2].y, h_pos.data[2].z);
        printf(" Particle 4: x = %f  y = %f  z = %f \n", h_pos.data[3].x, h_pos.data[3].y, h_pos.data[3].z);
        printf("\n");
    */
    
    // create the dihedral force compute to check
    shared_ptr<HarmonicDihedralForceCompute> fc_4 = tf_creator(sysdef_4);
    fc_4->setParams(0, Scalar(30.0), -1, 3); // type=0, K=30.0,sign=-1,multiplicity=3
    
    // compute the force and check the results
    fc_4->compute(0);
    
    {
    GPUArray<Scalar4>& force_array_1 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_1 =  fc_4->getVirialArray();
    unsigned int pitch = 0;
    ArrayHandle<Scalar4> h_force_1(force_array_1,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_1(virial_array_1,access_location::host,access_mode::read);
    
    // check that the force is correct, it should be 0 since we haven't created any dihedrals yet
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].y, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].z, tol);
    MY_BOOST_CHECK_SMALL(h_force_1.data[0].w, tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[0*pitch], tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[1*pitch], tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[2*pitch], tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[3*pitch], tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[4*pitch], tol);
    MY_BOOST_CHECK_SMALL(h_virial_1.data[5*pitch], tol);
    }

    // add an dihedrals and check again
    sysdef_4->getDihedralData()->addDihedral(Dihedral(0,0,1,2,3)); // add type 0 dihedral bewtween atoms 0-1-2-3
    fc_4->compute(1);
    
    {
    // this time there should be a force
    GPUArray<Scalar4>& force_array_2 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_2 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_2.getPitch();
    ArrayHandle<Scalar4> h_force_2(force_array_2,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_2(virial_array_2,access_location::host,access_mode::read);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].x, -0.5*0.118121, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].y, 0.5*0.856380, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].z, 0.5*1.063092, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[0].w, 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0*pitch+0]
                        +h_virial_2.data[3*pitch+0]
                        +h_virial_2.data[5*pitch+0], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].x, -0.5*0.522868, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].y, -0.5*0.525225, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].z, -0.5*0.226780, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[1].w, 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0*pitch+1]
                        +h_virial_2.data[3*pitch+1]
                        +h_virial_2.data[5*pitch+1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].x, 0.5*0.445003, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].y, -0.5*0.328375, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].z, -0.5*0.693145, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[2].w, 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0*pitch+2]
                        +h_virial_2.data[3*pitch+2]
                        +h_virial_2.data[5*pitch+2], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].x, 0.5*0.195986, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].y, -0.5*0.002780, loose_tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].z, -0.5*0.143167, tol);
    MY_BOOST_CHECK_CLOSE(h_force_2.data[3].w, 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(h_virial_2.data[0*pitch+3]
                        +h_virial_2.data[3*pitch+3]
                        +h_virial_2.data[5*pitch+3], tol);
    }

    /*
     FORCE 1: fx = -0.118121  fy = 0.856380  fz = 1.063092
     FORCE 2: fx = -0.522868  fy = -0.525225  fz = -0.226780
     FORCE 3: fx = 0.445003  fy = -0.328375  fz = -0.693145
     FORCE 4: fx = 0.195986  fy = -0.002780  fz = -0.143167
    
    */
    
    // rearrange the two particles in memory and see if they are properly updated
    {
    ArrayHandle<Scalar4> h_pos(pdata_4->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_4->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_4->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[1].x = Scalar(10.0); // put atom b at (10,1,2)
    h_pos.data[1].y = Scalar(1.0);
    h_pos.data[1].z = Scalar(2.0);

    h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = Scalar(1.0); // put atom a at (1,1,1)

    h_tag.data[0] = 1;
    h_tag.data[1] = 0;
    h_rtag.data[0] = 1;
    h_rtag.data[1] = 0;
    }

    // notify that we made the sort
    pdata_4->notifyParticleSort();
    // recompute at the same timestep, the forces should still be updated
    fc_4->compute(1);
    
    {
    GPUArray<Scalar4>& force_array_3 =  fc_4->getForceArray();
    GPUArray<Scalar>& virial_array_3 =  fc_4->getVirialArray();
    unsigned int pitch = virial_array_3.getPitch();
    ArrayHandle<Scalar4> h_force_3(force_array_3,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_3(virial_array_3,access_location::host,access_mode::read);
    
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].x, -0.5*0.118121, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].y, 0.5*0.856380, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].z, 0.5*1.063092, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[1].w, 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(h_virial_3.data[0*pitch+1]
                        +h_virial_3.data[3*pitch+1]
                        +h_virial_3.data[5*pitch+1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].x, -0.5*0.522868, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].y, -0.5*0.525225, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].z, -0.5*0.226780, tol);
    MY_BOOST_CHECK_CLOSE(h_force_3.data[0].w, 0.5*14.945559, tol);
    MY_BOOST_CHECK_SMALL(h_virial_3.data[0*pitch+0]
                        +h_virial_3.data[3*pitch+0]
                        +h_virial_3.data[5*pitch+0], tol);
    }
    
    ////////////////////////////////////////////////////////////////////
    // now, lets do a more thorough test and include boundary conditions
    // there are way too many permutations to test here, so I will simply
    // test +x, -x, +y, -y, +z, and -z independantly
    // build a 8 particle system with particles across each boundary
    // also test more than one type of dihedral
    shared_ptr<SystemDefinition> sysdef_8(new SystemDefinition(8, BoxDim(60.0, 70.0, 80.0), 1, 0, 0, 2, 0, exec_conf));
    shared_ptr<ParticleData> pdata_8 = sysdef_8->getParticleData();

    {
    ArrayHandle<Scalar4> h_pos(pdata_8->getPositions(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = -9.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 1.0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 10.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6);
    h_pos.data[5].x = 0; h_pos.data[5].y = 0; h_pos.data[5].z =  Scalar(29.6);
    h_pos.data[6].x = 3; h_pos.data[6].y = 3; h_pos.data[6].z =  Scalar(29.6);
    h_pos.data[7].x = 3; h_pos.data[7].y = 0; h_pos.data[7].z =  Scalar(31.0);
    }

    shared_ptr<HarmonicDihedralForceCompute> fc_8 = tf_creator(sysdef_8);
    fc_8->setParams(0, 50.0, -1, 3);
    fc_8->setParams(1, 30.0,  1, 4);
    
    sysdef_8->getDihedralData()->addDihedral(Dihedral(0, 0,1,2,3));
    sysdef_8->getDihedralData()->addDihedral(Dihedral(1, 4,5,6,7));
    
    fc_8->compute(0);
    
    {
    // check that the forces are correctly computed
    GPUArray<Scalar4>& force_array_4 =  fc_8->getForceArray();
    GPUArray<Scalar>& virial_array_4 =  fc_8->getVirialArray();
    unsigned int pitch = virial_array_4.getPitch();
    ArrayHandle<Scalar4> h_force_4(force_array_4,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_4(virial_array_4,access_location::host,access_mode::read);
    
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].x, tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[0].y, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].z, 0.5*10.504562,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[0].w, 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+0]
                        +h_virial_4.data[3*pitch+0]
                        +h_virial_4.data[5*pitch+0], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].x, -0.5*5.724548, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].y, 0.5*2.667751,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].z, 0.5*5.650116,tol); // 5.650116
    MY_BOOST_CHECK_CLOSE(h_force_4.data[1].w, 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+1]
                        +h_virial_4.data[3*pitch+1]
                        +h_virial_4.data[5*pitch+1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].x, 0.5*10.384811,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].y, -0.5*4.839524, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].z, -0.5*22.843740,tol); //
    MY_BOOST_CHECK_CLOSE(h_force_4.data[2].w, 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+2]
                        +h_virial_4.data[3*pitch+2]
                        +h_virial_4.data[5*pitch+2], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].x, -0.5*4.660264,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].y, 0.5*2.171773, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].z, 0.5*6.689062,tol); //
    MY_BOOST_CHECK_CLOSE(h_force_4.data[3].w, 0.5*7.687906, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+3]
                        +h_virial_4.data[3*pitch+3]
                        +h_virial_4.data[5*pitch+3], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[4].x, 0.5*2.949620,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[4].y, -0.5*2.949620,tol);
    MY_BOOST_CHECK_SMALL(h_force_4.data[4].z, tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[4].w, 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+4]
                        +h_virial_4.data[3*pitch+4]
                        +h_virial_4.data[5*pitch+4], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].x, 0.5*3.698439,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].y, -0.5*3.698439,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].z, -0.5*14.245843,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[5].w, 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+5]
                        +h_virial_4.data[3*pitch+5]
                        +h_virial_4.data[5*pitch+5], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].x, 0.5*6.648058,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].y, -0.5*6.648058,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].z, -0.5*14.245843,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[6].w, 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+6]
                        +h_virial_4.data[3*pitch+6]
                        +h_virial_4.data[5*pitch+6], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].x, -0.5*13.296117,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].y, 0.5*13.296117,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].z, 0.5*28.491686,tol);
    MY_BOOST_CHECK_CLOSE(h_force_4.data[7].w, 0.5*2.318964, tol);
    MY_BOOST_CHECK_SMALL(h_virial_4.data[0*pitch+7]
                        +h_virial_4.data[3*pitch+7]
                        +h_virial_4.data[5*pitch+7], tol);
    }
    
    // one more test: this one will test two things:
    // 1) That the forces are computed correctly even if the particles are rearranged in memory
    // and 2) That two forces can add to the same particle
    shared_ptr<SystemDefinition> sysdef_5(new SystemDefinition(5, BoxDim(100.0, 100.0, 100.0), 1, 0, 0, 1, 0, exec_conf));
    shared_ptr<ParticleData> pdata_5 = sysdef_5->getParticleData();
    
    {
    ArrayHandle<Scalar4> h_pos(pdata_5->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(pdata_5->getTags(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(pdata_5->getRTags(), access_location::host, access_mode::readwrite);

    h_pos.data[0].x = Scalar(-9.6); h_pos.data[0].y = -9.0; h_pos.data[0].z = 0.0;
    h_pos.data[1].x =  Scalar(9.6); h_pos.data[1].y = 1.0; h_pos.data[1].z = 0.0;
    h_pos.data[2].x = 0; h_pos.data[2].y = Scalar(-19.6); h_pos.data[2].z = 0.0;
    h_pos.data[3].x = 0; h_pos.data[3].y = Scalar(19.6); h_pos.data[3].z = 10.0;
    h_pos.data[4].x = 0; h_pos.data[4].y = 0; h_pos.data[4].z = Scalar(-29.6);

    h_tag.data[0] = 2;
    h_tag.data[1] = 3;
    h_tag.data[2] = 0;
    h_tag.data[3] = 1;
    h_rtag.data[h_tag.data[0]] = 0;
    h_rtag.data[h_tag.data[1]] = 1;
    h_rtag.data[h_tag.data[2]] = 2;
    h_rtag.data[h_tag.data[3]] = 3;
    }
    
    // build the dihedral force compute and try it out
    shared_ptr<HarmonicDihedralForceCompute> fc_5 = tf_creator(sysdef_5);
    fc_5->setParams(0, 15.0, -1, 4);
    
    sysdef_5->getDihedralData()->addDihedral(Dihedral(0, 0,1,2,3));
    sysdef_5->getDihedralData()->addDihedral(Dihedral(0, 1,2,3,4));
    
    fc_5->compute(0);
    
    {
    GPUArray<Scalar4>& force_array_5 =  fc_5->getForceArray();
    GPUArray<Scalar>& virial_array_5 =  fc_5->getVirialArray();
    unsigned int pitch = virial_array_5.getPitch();
    ArrayHandle<Scalar4> h_force_5(force_array_5,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_5(virial_array_5,access_location::host,access_mode::read);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].x, -0.5*1.400442, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].y, -0.5*1.251086,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].z, 0.5*3.152951,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[0].w, 0.5*7.944149, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+0]
                        +h_virial_5.data[3*pitch+0]
                        +h_virial_5.data[5*pitch+0], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].x, 0.5*1.719594, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].y, -0.5*3.301620,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].z, 0.5*5.293722,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[1].w, 0.5*7.944149, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+1]
                        +h_virial_5.data[3*pitch+1]
                        +h_virial_5.data[5*pitch+1], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].x, 0.5*1.153410, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].y, 0.5*1.044598,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].z, -0.5*4.094823,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[2].w, 0.5*5.176867, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+2]
                        +h_virial_5.data[3*pitch+2]
                        +h_virial_5.data[5*pitch+2], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].x, -0.5*0.581728, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].y, 0.5*1.797707,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].z, -0.5*4.582985,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[3].w, 0.5*7.944149, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+3]
                        +h_virial_5.data[3*pitch+3]
                        +h_virial_5.data[5*pitch+3], tol);
    
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].x, -0.5*0.890834, tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].y, 0.5*1.710401,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].z, 0.5*0.231135,tol);
    MY_BOOST_CHECK_CLOSE(h_force_5.data[4].w, 0.5*2.767281, tol);
    MY_BOOST_CHECK_SMALL(h_virial_5.data[0*pitch+4]
                        +h_virial_5.data[3*pitch+4]
                        +h_virial_5.data[5*pitch+4], tol);
    }
    }

//! Compares the output of two HarmonicDihedralForceComputes
void dihedral_force_comparison_tests(dihedralforce_creator tf_creator1,
                                     dihedralforce_creator tf_creator2,
                                     boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    const unsigned int N = 1000;
    
    // create a particle system to sum forces on
    // just randomly place particles. We don't really care how huge the bond forces get: this is just a unit test
    RandomInitializer rand_init(N, Scalar(0.2), Scalar(0.9), "A");
    SnapshotSystemData snap;
    rand_init.initSnapshot(snap);
    snap.dihedral_data.type_mapping.push_back("A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(snap, exec_conf));
    
    shared_ptr<HarmonicDihedralForceCompute> fc1 = tf_creator1(sysdef);
    shared_ptr<HarmonicDihedralForceCompute> fc2 = tf_creator2(sysdef);
    fc1->setParams(0, Scalar(3.0), -1, 3);
    fc2->setParams(0, Scalar(3.0), -1, 3);
    
    // add dihedrals
    for (unsigned int i = 0; i < N-3; i++)
        {
        sysdef->getDihedralData()->addDihedral(Dihedral(0, i, i+1,i+2, i+3));
        }
        
    // compute the forces
    fc1->compute(0);
    fc2->compute(0);
    
    // verify that the forces are identical (within roundoff errors)
    {
    GPUArray<Scalar4>& force_array_7 =  fc1->getForceArray();
    GPUArray<Scalar>& virial_array_7 =  fc1->getVirialArray();
    unsigned int pitch = virial_array_7.getPitch();
    ArrayHandle<Scalar4> h_force_7(force_array_7,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_7(virial_array_7,access_location::host,access_mode::read);
    GPUArray<Scalar4>& force_array_8 =  fc2->getForceArray();
    GPUArray<Scalar>& virial_array_8 =  fc2->getVirialArray();
    ArrayHandle<Scalar4> h_force_8(force_array_8,access_location::host,access_mode::read);
    ArrayHandle<Scalar> h_virial_8(virial_array_8,access_location::host,access_mode::read);
    
    // compare average deviation between the two computes
    double deltaf2 = 0.0;
    double deltape2 = 0.0;
    double deltav2[6];
    for (unsigned int i = 0; i < 6; i++)
        deltav2[i] = 0.0;
        
    for (unsigned int i = 0; i < N; i++)
        {
        deltaf2 += double(h_force_8.data[i].x - h_force_7.data[i].x) * double(h_force_8.data[i].x - h_force_7.data[i].x);
        deltaf2 += double(h_force_8.data[i].y - h_force_7.data[i].y) * double(h_force_8.data[i].y - h_force_7.data[i].y);
        deltaf2 += double(h_force_8.data[i].z - h_force_7.data[i].z) * double(h_force_8.data[i].z - h_force_7.data[i].z);
        deltape2 += double(h_force_8.data[i].w - h_force_7.data[i].w) * double(h_force_8.data[i].w - h_force_7.data[i].w);
        for (unsigned int j = 0; j < 6; j++)
            deltav2[j] += double(h_virial_8.data[j*pitch+i] - h_virial_7.data[j*pitch+i]) * double(h_virial_8.data[j*pitch+i] - h_virial_7.data[j*pitch+i]);

        // also check that each individual calculation is somewhat close
        }
    deltaf2 /= double(N);
    deltape2 /= double(N);
    for (unsigned int j = 0; j < 6; j++)
        deltav2[j] /= double(N);

    BOOST_CHECK_SMALL(deltaf2, double(tol_small));
    BOOST_CHECK_SMALL(deltape2, double(tol_small));
    BOOST_CHECK_SMALL(deltav2[0], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[1], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[2], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[3], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[4], double(tol_small));
    BOOST_CHECK_SMALL(deltav2[5], double(tol_small));
    }   
    }

//! HarmonicDihedralForceCompute creator for dihedral_force_basic_tests()
shared_ptr<HarmonicDihedralForceCompute> base_class_tf_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<HarmonicDihedralForceCompute>(new HarmonicDihedralForceCompute(sysdef));
    }

#ifdef ENABLE_CUDA
//! DihedralForceCompute creator for bond_force_basic_tests()
shared_ptr<HarmonicDihedralForceCompute> gpu_tf_creator(shared_ptr<SystemDefinition> sysdef)
    {
    return shared_ptr<HarmonicDihedralForceCompute>(new HarmonicDihedralForceComputeGPU(sysdef));
    }
#endif

//! boost test case for dihedral forces on the CPU
BOOST_AUTO_TEST_CASE( HarmonicDihedralForceCompute_basic )
    {
    printf(" IN BOOST_AUTO_TEST_CASE: CPU \n");
    dihedralforce_creator tf_creator = bind(base_class_tf_creator, _1);
    dihedral_force_basic_tests(tf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! boost test case for dihedral forces on the GPU
BOOST_AUTO_TEST_CASE( HarmonicDihedralForceComputeGPU_basic )
    {
    printf(" IN BOOST_AUTO_TEST_CASE: GPU \n");
    dihedralforce_creator tf_creator = bind(gpu_tf_creator, _1);
    dihedral_force_basic_tests(tf_creator, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing bond GPU and CPU BondForceComputes
BOOST_AUTO_TEST_CASE( HarmonicDihedralForceComputeGPU_compare )
    {
    dihedralforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    dihedralforce_creator tf_creator = bind(base_class_tf_creator, _1);
    dihedral_force_comparison_tests(tf_creator, tf_creator_gpu, boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }

//! boost test case for comparing calculation on the CPU to multi-gpu ones
BOOST_AUTO_TEST_CASE( HarmonicDihedralForce_MultiGPU_compare)
    {
    boost::shared_ptr<ExecutionConfiguration> exec_conf(new ExecutionConfiguration(ExecutionConfiguration::GPU));
    
    dihedralforce_creator tf_creator_gpu = bind(gpu_tf_creator, _1);
    dihedralforce_creator tf_creator = bind(base_class_tf_creator, _1);
    dihedral_force_comparison_tests(tf_creator, tf_creator_gpu, exec_conf);
    }
#endif


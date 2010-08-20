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
// Maintainer: joaander

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

#include "NeighborList.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"

#include <sstream>
#include <fstream>

#include <iostream>
#include <stdexcept>

using namespace boost;
using namespace std;

/*! \file NeighborList.cc
    \brief Defines the NeighborList class
*/

/*! \param sysdef System the neighborlist is to compute neighbors for
    \param r_cut Cuttoff radius under which particles are considered neighbors
    \param r_buff Buffere radius around \a r_cut in which neighbors will be included

    \post NeighborList is initialized and the list memory has been allocated,
        but the list will not be computed until compute is called.
    \post The storage mode defaults to half
*/
NeighborList::NeighborList(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff)
    : Compute(sysdef), m_r_cut(r_cut), m_r_buff(r_buff), m_storage_mode(half), m_updates(0), m_forced_updates(0), 
      m_dangerous_updates(0), m_force_update(true)
    {
    // check for two sensless errors the user could make
    if (m_r_cut < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error initializing NeighborList");
        }
        
    if (m_r_buff < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error initializing NeighborList");
        }
        
    // initialize values
    m_last_updated_tstep = 0;
    m_every = 0;
    m_Nmax = 256;
    
    // allocate m_n_neigh and m_last_pos
    GPUArray<unsigned int> n_neigh(m_pdata->getN(), exec_conf);
    m_n_neigh.swap(n_neigh);
    GPUArray<Scalar4> last_pos(m_pdata->getN(), exec_conf);
    m_last_pos.swap(last_pos);
    
    // allocate nlist array
    allocateNlist();
    
    m_sort_connection = m_pdata->connectParticleSort(bind(&NeighborList::forceUpdate, this));
    }

NeighborList::~NeighborList()
    {
    m_sort_connection.disconnect();
    }

/*! Updates the neighborlist if it has not yet been updated this times step
    \param timestep Current time step of the simulation
*/
void NeighborList::compute(unsigned int timestep)
    {
    // skip if we shouldn't compute this step
    if (!shouldCompute(timestep) && !m_force_update)
        return;
        
    if (m_prof) m_prof->push("Neighbor");
    
#ifdef ENABLE_CUDA
    // update the exclusion data if this is a forced update
    //if (m_force_update)
        //updateExclusionData();
#endif
        
    // check if the list needs to be updated and update it
    if (needsUpdating(timestep))
        {
        buildNlist();
        setLastUpdatedPos();
        }
        
    if (m_prof) m_prof->pop();
    }

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation

    Calls buildNlist repeatedly to benchmark the neighbor list.
*/
double NeighborList::benchmark(unsigned int num_iters)
    {
    ClockSource t;
    // warm up run
    forceUpdate();
    compute(0);
    buildNlist();
    
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        }
#endif
    
    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        buildNlist();
        
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        cudaThreadSynchronize();
#endif
    uint64_t total_time_ns = t.getTime() - start_time;
    
    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

/*! \param r_cut New cuttoff radius to set
    \param r_buff New buffer radius to set
    \note Changing the cuttoff radius does NOT immeadiately update the neighborlist.
            The new cuttoff will take effect when compute is called for the next timestep.
*/
void NeighborList::setRCut(Scalar r_cut, Scalar r_buff)
    {
    m_r_cut = r_cut;
    m_r_buff = r_buff;
    
    // check for two sensless errors the user could make
    if (m_r_cut < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error changing NeighborList parameters");
        }
        
    if (m_r_buff < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error changing NeighborList parameters");
        }
        
    forceUpdate();
    }

/*! \returns an estimate of the number of neighbors per particle
    This mean-field estimate may be very bad dending on how clustered particles are.
    Derived classes can override this method to provide better estimates.

    \note Under NO circumstances should calling this method produce any
    appreciable amount of overhead. This is mainly a warning to
    derived classes.
*/
Scalar NeighborList::estimateNNeigh()
    {
    // calculate a number density of particles
    BoxDim box = m_pdata->getBox();
    Scalar vol = (box.xhi - box.xlo)*(box.yhi - box.ylo)*(box.zhi - box.zlo);
    Scalar n_dens = Scalar(m_pdata->getN()) / vol;
    
    // calculate the average number of neighbors by multiplying by the volume
    // within the cutoff
    Scalar r_max = m_r_cut + m_r_buff;
    Scalar vol_cut = Scalar(4.0/3.0 * M_PI) * r_max * r_max * r_max;
    return n_dens * vol_cut;
    }

/*! \param tag1 TAG (not index) of the first particle in the pair
    \param tag2 TAG (not index) of the second particle in the pair
    \post The pair \a tag1, \a tag2 will not appear in the neighborlist
    \note This only takes effect on the next call to compute() that updates the list
    \note Only 4 particles can be excluded from a single particle's neighbor list,
    \note unless the code is compiled with the option LARGE_EXCLUSION_LIST, in which
    \note case the maximum is 16 exclusions. Duplicates are checked for and not added.
*/
void NeighborList::addExclusion(unsigned int tag1, unsigned int tag2)
    {
    /*if (tag1 >= m_pdata->getN() || tag2 >= m_pdata->getN())
        {
        cerr << endl << "***Error! Particle tag out of bounds when attempting to add neighborlist exclusion: " << tag1 << "," << tag2 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
    // don't add an exclusion twice and waste space in the memory restricted exclusion lists.
    if (isExcluded(tag1, tag2))
        return;
        
    // add tag2 to tag1's exculsion list
    if (m_exclusions[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e1 = tag2;
    else if (m_exclusions[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e2 = tag2;
    else if (m_exclusions[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e3 = tag2;
    else if (m_exclusions[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e4 = tag2;
#if defined(LARGE_EXCLUSION_LIST)
    else if (m_exclusions2[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e1 = tag2;
    else if (m_exclusions2[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e2 = tag2;
    else if (m_exclusions2[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e3 = tag2;
    else if (m_exclusions2[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e4 = tag2;
    else if (m_exclusions3[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e1 = tag2;
    else if (m_exclusions3[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e2 = tag2;
    else if (m_exclusions3[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e3 = tag2;
    else if (m_exclusions3[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e4 = tag2;
    else if (m_exclusions4[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e1 = tag2;
    else if (m_exclusions4[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e2 = tag2;
    else if (m_exclusions4[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e3 = tag2;
    else if (m_exclusions4[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e4 = tag2;
#endif
    else
        {
        // error: exclusion list full
        cerr << endl << "***Error! Exclusion list full for particle with tag: " << tag1 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
        
    // add tag1 to tag2's exclusion list
    if (m_exclusions[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e1 = tag1;
    else if (m_exclusions[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e2 = tag1;
    else if (m_exclusions[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e3 = tag1;
    else if (m_exclusions[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e4 = tag1;
#if defined(LARGE_EXCLUSION_LIST)
    else if (m_exclusions2[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e1 = tag1;
    else if (m_exclusions2[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e2 = tag1;
    else if (m_exclusions2[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e3 = tag1;
    else if (m_exclusions2[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e4 = tag1;
    else if (m_exclusions3[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e1 = tag1;
    else if (m_exclusions3[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e2 = tag1;
    else if (m_exclusions3[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e3 = tag1;
    else if (m_exclusions3[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e4 = tag1;
    else if (m_exclusions4[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e1 = tag1;
    else if (m_exclusions4[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e2 = tag1;
    else if (m_exclusions4[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e3 = tag1;
    else if (m_exclusions4[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e4 = tag1;
#endif
    else
        {
        // error: exclusion list full
        cerr << endl << "***Error! Exclusion list full for particle with tag: " << tag2 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
    forceUpdate();*/
    }

/*! \post No particles are excluded from the neighbor list
*/
void NeighborList::clearExclusions()
    {
    /*for (unsigned int i = 0; i < m_exclusions.size(); i++)
        {
        m_exclusions[i].e1 = EXCLUDE_EMPTY;
        m_exclusions[i].e2 = EXCLUDE_EMPTY;
        m_exclusions[i].e3 = EXCLUDE_EMPTY;
        m_exclusions[i].e4 = EXCLUDE_EMPTY;
        }
#if defined(LARGE_EXCLUSION_LIST)
    for (unsigned int i = 0; i < m_exclusions2.size(); i++)
        {
        m_exclusions2[i].e1 = EXCLUDE_EMPTY;
        m_exclusions2[i].e2 = EXCLUDE_EMPTY;
        m_exclusions2[i].e3 = EXCLUDE_EMPTY;
        m_exclusions2[i].e4 = EXCLUDE_EMPTY;
        }
    for (unsigned int i = 0; i < m_exclusions3.size(); i++)
        {
        m_exclusions3[i].e1 = EXCLUDE_EMPTY;
        m_exclusions3[i].e2 = EXCLUDE_EMPTY;
        m_exclusions3[i].e3 = EXCLUDE_EMPTY;
        m_exclusions3[i].e4 = EXCLUDE_EMPTY;
        }
    for (unsigned int i = 0; i < m_exclusions4.size(); i++)
        {
        m_exclusions4[i].e1 = EXCLUDE_EMPTY;
        m_exclusions4[i].e2 = EXCLUDE_EMPTY;
        m_exclusions4[i].e3 = EXCLUDE_EMPTY;
        m_exclusions4[i].e4 = EXCLUDE_EMPTY;
        }
#endif
    forceUpdate();*/
    }

/*! \post Gather some statistics about exclusions usage.
*/
void NeighborList::countExclusions()
    {
    /*unsigned int excluded_count[MAX_NUM_EXCLUDED+1];
    unsigned int num_excluded, max_num_excluded;
    
    max_num_excluded = 0;
    for (unsigned int c=0; c <= MAX_NUM_EXCLUDED; ++c)
        excluded_count[c] = 0;
        
#if !defined(LARGE_EXCLUSION_LIST)
    for (unsigned int i = 0; i < m_exclusions.size(); i++)
        {
        num_excluded = 0;
        if (m_exclusions[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (num_excluded > max_num_excluded) max_num_excluded = num_excluded;
        excluded_count[num_excluded] += 1;
        }
#else
    for (unsigned int i = 0; i < m_exclusions.size(); i++)
        {
        num_excluded = 0;
        if (m_exclusions[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (num_excluded > max_num_excluded) max_num_excluded = num_excluded;
        excluded_count[num_excluded] += 1;
        }
#endif
    cout << "-- Neighborlist exclusion statistics:" << endl;
    cout << "Max. number of exclusions: " << max_num_excluded << endl;
    for (unsigned int i=0; i <= max_num_excluded; ++i)
        cout << "Particles with " << i << " exclusions: " << excluded_count[i] << endl;*/
    }

/*! After calling addExclusionFromBonds() all bonds specified in the attached ParticleData will be
    added as exlusions. Any additional bonds added after this will not be automatically added as exclusions.
*/
void NeighborList::addExclusionsFromBonds()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    
    // for each bond
    for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
        {
        // add an exclusion
        Bond bond = bond_data->getBond(i);
        addExclusion(bond.a, bond.b);
        }
    }

/*! After calling addExclusionsFromAngles(), all angles specified in the attached ParticleData will be added to the
    exclusion list. Only the two end particles in the angle are excluded from interacting.
*/
void NeighborList::addExclusionsFromAngles()
    {
    boost::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();
    
    // for each bond
    for (unsigned int i = 0; i < angle_data->getNumAngles(); i++)
        {
        Angle angle = angle_data->getAngle(i);
        addExclusion(angle.a, angle.c);
        }
    }

/*! After calling addExclusionsFromAngles(), all dihedrals specified in the attached ParticleData will be added to the
    exclusion list. Only the two end particles in the dihedral are excluded from interacting.
*/
void NeighborList::addExclusionsFromDihedrals()
    {
    boost::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();
    
    // for each bond
    for (unsigned int i = 0; i < dihedral_data->getNumDihedrals(); i++)
        {
        Dihedral dihedral = dihedral_data->getDihedral(i);
        addExclusion(dihedral.a, dihedral.d);
        }
    }

/*! \param tag1 First particle tag in the pair
    \param tag2 Second particle tag in the pair
    \return true if the particles \a tag1 and \a tag2 have been excluded from the neighbor list
*/
bool NeighborList::isExcluded(unsigned int tag1, unsigned int tag2)
    {
    /*if (tag1 >= m_pdata->getN() || tag2 >= m_pdata->getN())
        {
        cerr << endl << "***Error! Particle tag out of bounds when attempting to add neighborlist exclusion: " << tag1 << "," << tag2 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
        
    if (m_exclusions[tag1].e1 == tag2)
        return true;
    if (m_exclusions[tag1].e2 == tag2)
        return true;
    if (m_exclusions[tag1].e3 == tag2)
        return true;
    if (m_exclusions[tag1].e4 == tag2)
        return true;
#if defined(LARGE_EXCLUSION_LIST)
    if (m_exclusions2[tag1].e1 == tag2)
        return true;
    if (m_exclusions2[tag1].e2 == tag2)
        return true;
    if (m_exclusions2[tag1].e3 == tag2)
        return true;
    if (m_exclusions2[tag1].e4 == tag2)
        return true;
    if (m_exclusions3[tag1].e1 == tag2)
        return true;
    if (m_exclusions3[tag1].e2 == tag2)
        return true;
    if (m_exclusions3[tag1].e3 == tag2)
        return true;
    if (m_exclusions3[tag1].e4 == tag2)
        return true;
    if (m_exclusions4[tag1].e1 == tag2)
        return true;
    if (m_exclusions4[tag1].e2 == tag2)
        return true;
    if (m_exclusions4[tag1].e3 == tag2)
        return true;
    if (m_exclusions4[tag1].e4 == tag2)
        return true;
#endif
    return false;*/
    }

/*! Add topologically derived exclusions for angles
 *
 * This excludes all non-bonded interactions between all pairs particles
 * that are bonded to the same atom.
 * To make the process quasi-linear scaling with system size we first
 * create a 1-d array the collects the number and index of bond partners.
 */
void NeighborList::addOneThreeExclusionsFromTopology()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = m_pdata->getN();
    const unsigned int MAXNBONDS = 7+1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNumBonds();
    
    if (nBonds == 0)
        {
        cout << "***Warning! No bonds defined while trying to add topology derived 1-3 exclusions" << endl;
        return;
        }
        
    // build a per atom list with all bonding partners from the list of bonds.
    unsigned int *localBondList = new unsigned int[MAXNBONDS*myNAtoms];
    memset((void *)localBondList,0,sizeof(unsigned int)*MAXNBONDS*myNAtoms);
    
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exlcusion map
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        // next, incrememt the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB*MAXNBONDS];
        
        if (nBondsA >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagA << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        if (nBondsB >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagB << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        localBondList[tagA*MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB*MAXNBONDS + nBondsB] = tagA;
        }
        
    // now loop over the atoms and build exclusions if we have more than
    // one bonding partner, i.e. we are in the center of an angle.
    for (unsigned int i = 0; i < myNAtoms; i++)
        {
        // now, loop over all atoms, and find those in the middle of an angle
        const unsigned int iAtom = i*MAXNBONDS;
        const unsigned int nBonds = localBondList[iAtom];
        
        if (nBonds > 1) // need at least two bonds
            {
            for (unsigned int j = 1; j < nBonds; ++j)
                {
                for (unsigned int k = j+1; k <= nBonds; ++k)
                    addExclusion(localBondList[iAtom+j],localBondList[iAtom+k]);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }

/*! Add topologically derived exclusions for dihedrals
 *
 * This excludes all non-bonded interactions between all pairs particles
 * that are connected to a common bond.
 *
 * To make the process quasi-linear scaling with system size we first
 * create a 1-d array the collects the number and index of bond partners.
 * and then loop over bonded partners.
 */
void NeighborList::addOneFourExclusionsFromTopology()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = m_pdata->getN();
    const unsigned int MAXNBONDS = 7+1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNumBonds();
    
    if (nBonds == 0)
        {
        cout << "***Warning! No bonds defined while trying to add topology derived 1-4 exclusions" << endl;
        return;
        }
        
    // allocate and clear data.
    unsigned int *localBondList = new unsigned int[MAXNBONDS*myNAtoms];
    memset((void *)localBondList,0,sizeof(unsigned int)*MAXNBONDS*myNAtoms);
    
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exlcusion map
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        // next, incrememt the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB*MAXNBONDS];
        
        if (nBondsA >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagA << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        if (nBondsB >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagB << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        localBondList[tagA*MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB*MAXNBONDS + nBondsB] = tagA;
        }
        
    //  loop over all bonds
    for (unsigned int i = 0; i < nBonds; i++)
        {
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        const unsigned int nBondsA = localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = localBondList[tagB*MAXNBONDS];
        
        for (unsigned int j = 1; j <= nBondsA; j++)
            {
            const unsigned int tagJ = localBondList[tagA*MAXNBONDS+j];
            if (tagJ == tagB) // skip the bond in the middle of the dihedral
                continue;
                
            for (unsigned int k = 1; k <= nBondsB; k++)
                {
                const unsigned int tagK = localBondList[tagB*MAXNBONDS+k];
                if (tagK == tagA) // skip the bond in the middle of the dihedral
                    continue;
                    
                addExclusion(tagJ,tagK);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }


/*! \returns true If any of the particles have been moved more than 1/2 of the buffer distance since the last call
        to this method that returned true.
    \returns false If none of the particles has been moved more than 1/2 of the buffer distance since the last call to this
        method that returned true.

    Note: this method relies on data set by setLastUpdatedPos(), which must be called to set the previous data used
    in the next call to distanceCheck();
*/
bool NeighborList::distanceCheck()
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // profile
    if (m_prof) m_prof->push("Dist check");
    
    // temporary storage for the result
    bool result = false;
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::read);
    
    // actually scan the array looking for values over 1/2 the buffer distance
    Scalar maxsq = (m_r_buff/Scalar(2.0))*(m_r_buff/Scalar(2.0));
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        Scalar dx = arrays.x[i] - h_last_pos.data[i].x;
        Scalar dy = arrays.y[i] - h_last_pos.data[i].y;
        Scalar dz = arrays.z[i] - h_last_pos.data[i].z;
        
        // if the vector crosses the box, pull it back
        if (dx >= box.xhi)
            dx -= Lx;
        else if (dx < box.xlo)
            dx += Lx;
            
        if (dy >= box.yhi)
            dy -= Ly;
        else if (dy < box.ylo)
            dy += Ly;
            
        if (dz >= box.zhi)
            dz -= Lz;
        else if (dz < box.zlo)
            dz += Lz;
            
        if (dx*dx + dy*dy + dz*dz >= maxsq)
            {
            result = true;
            break;
            }
        }
        
    // don't worry about computing flops here, this is fast
    if (m_prof) m_prof->pop();
    
    m_pdata->release();
    return result;
    }

/*! Copies the current positions of all particles over to m_last_x etc...
*/
void NeighborList::setLastUpdatedPos()
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // profile
    if (m_prof) m_prof->push("Dist check");
    
    // update the last position arrays
    ArrayHandle<Scalar4> h_last_pos(m_last_pos, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        h_last_pos.data[i] = make_scalar4(arrays.x[i], arrays.y[i], arrays.z[i], 0.0f);
        }
    
    if (m_prof) m_prof->pop();
    m_pdata->release();
    }

/*! \returns true If the neighbor list needs to be updated
    \returns false If the neighbor list does not need to be updated
    \note This is designed to be called if (needsUpdating()) then update every step.
        It internally handles many state variables that rely on this assumption.
    \param timestep Current time step in the simulation
*/
bool NeighborList::needsUpdating(unsigned int timestep)
    {
    if (timestep < (m_last_updated_tstep + m_every) && !m_force_update)
        return false;
        
    // check if this is a dangerous time
    // we are dangerous if m_every is greater than 1 and this is the first check after the
    // last build
    bool dangerous = false;
    if (m_every > 1 && timestep == (m_last_updated_tstep + m_every))
        dangerous = true;
        
    // temporary storage for return result
    bool result = false;
    
    // if the update has been forced, the result defaults to true
    if (m_force_update)
        {
        result = true;
        m_force_update = false;
        m_forced_updates += 1;
        m_last_updated_tstep = timestep;
        
        // when an update is forced, there is no way to tell if the build
        // is dangerous or not: filter out the false positive errors
        dangerous = false;
        }
    else
        {
        // not a forced update, perform the distance check to determine
        // if the list needs to be updated - no dist check needed if r_buff is tiny
        if (m_r_buff < 1e-6)
            result = true;
        else
            result = distanceCheck();
        
        if (result)
            {
            m_last_updated_tstep = timestep;
            m_updates += 1;
            }
        }
        
    // warn the user if this is a dangerous build
    if (result && dangerous)
        {
        cout << "***Warning! Dangerous neighborlist build occured. Continuing this simulation may produce incorrect results and/or program crashes. Decrease the neighborlist check_period and rerun." << endl;
        m_dangerous_updates += 1;
        }
        
    return result;
    }

/*! Generic statistics that apply to any neighbor list, like the number of updates,
    average number of neighbors, etc... are printed to stdout. Derived classes should
    print any pertinient information they see fit to.
 */
void NeighborList::printStats()
    {
    cout << "-- Neighborlist stats:" << endl;
    cout << m_updates << " normal updates / " << m_forced_updates << " forced updates / " << m_dangerous_updates << " dangerous updates" << endl;
    
    // access the number of neighbors to generate stats
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::read);
    
    // build some simple statistics of the number of neighbors
    unsigned int n_neigh_min = m_pdata->getN();
    unsigned int n_neigh_max = 0;
    Scalar n_neigh_avg = 0.0;
    
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int n_neigh = (unsigned int)h_n_neigh.data[i];
        if (n_neigh < n_neigh_min)
            n_neigh_min = n_neigh;
        if (n_neigh > n_neigh_max)
            n_neigh_max = n_neigh;
            
        n_neigh_avg += Scalar(n_neigh);
        }
        
    // divide to get the average
    n_neigh_avg /= Scalar(m_pdata->getN());
    
    cout << "n_neigh_min: " << n_neigh_min << " / n_neigh_max: " << n_neigh_max << " / n_neigh_avg: " << n_neigh_avg << endl;
    }

void NeighborList::resetStats()
    {
    m_updates = m_forced_updates = m_dangerous_updates = 0;
    }

/*! Loops through the particles and finds all of the particles \c j who's distance is less than
    \c r_cut \c + \c r_buff from particle \c i, includes either i < j or all neighbors depending
    on the mode set by setStorageMode()
*/
void NeighborList::buildNlist()
    {
    // sanity check
    assert(m_pdata);
    
    // start up the profile
    if (m_prof) m_prof->push("Build list");
    
    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    if ((box.xhi - box.xlo) <= (m_r_cut+m_r_buff) * 2.0 || (box.yhi - box.ylo) <= (m_r_cut+m_r_buff) * 2.0 || (box.zhi - box.zlo) <= (m_r_cut+m_r_buff) * 2.0)
        {
        cerr << endl << "***Error! Simulation box is too small! Particles would be interacting with themselves." << endl << endl;
        throw runtime_error("Error updating neighborlist bins");
        }
        
    // access the nlist data
    ArrayHandle<unsigned int> h_n_neigh(m_n_neigh, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_nlist(m_nlist, access_location::host, access_mode::overwrite);
    
    // simple algorithm follows:
    
    // start by creating a temporary copy of r_cut sqaured
    Scalar rmaxsq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    Scalar Lx2 = Lx / Scalar(2.0);
    Scalar Ly2 = Ly / Scalar(2.0);
    Scalar Lz2 = Lz / Scalar(2.0);
    
    
    // start by clearing the entire list
    memset(h_n_neigh.data, 0, sizeof(unsigned int)*arrays.nparticles);
    
    // now we can loop over all particles in n^2 fashion and build the list
#pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < (int)arrays.nparticles; i++)
        {
        Scalar xi = arrays.x[i];
        Scalar yi = arrays.y[i];
        Scalar zi = arrays.z[i];
        
        // for each other particle with i < j
        for (unsigned int j = i + 1; j < arrays.nparticles; j++)
            {
            // calculate dr
            Scalar dx = arrays.x[j] - xi;
            Scalar dy = arrays.y[j] - yi;
            Scalar dz = arrays.z[j] - zi;
            
            // if the vector crosses the box, pull it back
            if (dx >= Lx2)
                dx -= Lx;
            else if (dx < -Lx2)
                dx += Lx;
                
            if (dy >= Ly2)
                dy -= Ly;
            else if (dy < -Ly2)
                dy += Ly;
                
            if (dz >= Lz2)
                dz -= Lz;
            else if (dz < -Lz2)
                dz += Lz;
                
            // sanity check
            assert(dx >= box.xlo && dx <= box.xhi);
            assert(dy >= box.ylo && dy <= box.yhi);
            assert(dz >= box.zlo && dz <= box.zhi);
            
            // now compare rsq to rmaxsq and add to the list if it meets the criteria
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < rmaxsq)
                {
                if (m_storage_mode == full)
                    {
                    #pragma omp critical
                        {
                        int posi = h_n_neigh.data[i];
                        h_nlist.data[m_nlist_indexer(i, posi)] = j;
                        h_n_neigh.data[i]++;
                        
                        int posj = h_n_neigh.data[j];
                        h_nlist.data[m_nlist_indexer(j, posj)] = i;
                        h_n_neigh.data[j]++;
                        }
                    }
                else
                    {
                    int pos = h_n_neigh.data[i];
                    h_nlist.data[m_nlist_indexer(i, pos)] = j;
                    h_n_neigh.data[i]++;
                    }
                }
            }
        }
        
    m_pdata->release();
    
    if (m_prof) m_prof->pop();
    }

void NeighborList::allocateNlist()
    {
    // allocate the memory
    GPUArray<unsigned int> nlist(m_pdata->getN(), m_Nmax+1, exec_conf);
    m_nlist.swap(nlist);
    
    // update the indexer
    m_nlist_indexer = Index2D(m_nlist.getPitch(), m_Nmax);
    }

//! helper function for accessing an elemeng of the neighb rlist: python __getitem__
/*! \param list List to extract an item from
    \param i item to extract
*/
unsigned int getNlistItem(std::vector<unsigned int>* list, unsigned int i)
    {
    return (*list)[i];
    }

void export_NeighborList()
    {
    class_< std::vector<unsigned int> >("std_vector_uint")
    .def("__len__", &std::vector<unsigned int>::size)
    .def("__getitem__", &getNlistItem)
    .def("push_back", &std::vector<unsigned int>::push_back)
    ;
    
    scope in_nlist = class_<NeighborList, boost::shared_ptr<NeighborList>, bases<Compute>, boost::noncopyable >
                     ("NeighborList", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     .def("setRCut", &NeighborList::setRCut)
                     .def("setEvery", &NeighborList::setEvery)
                     .def("setStorageMode", &NeighborList::setStorageMode)
                     .def("addExclusion", &NeighborList::addExclusion)
                     .def("clearExclusions", &NeighborList::clearExclusions)
                     .def("countExclusions", &NeighborList::countExclusions)
                     .def("addExclusionsFromBonds", &NeighborList::addExclusionsFromBonds)
                     .def("addExclusionsFromAngles", &NeighborList::addExclusionsFromAngles)
                     .def("addExclusionsFromDihedrals", &NeighborList::addExclusionsFromDihedrals)
                     .def("addOneThreeExclusionsFromTopology", &NeighborList::addOneThreeExclusionsFromTopology)
                     .def("addOneFourExclusionsFromTopology", &NeighborList::addOneFourExclusionsFromTopology)
                     .def("forceUpdate", &NeighborList::forceUpdate)
                     .def("estimateNNeigh", &NeighborList::estimateNNeigh)
                     ;
                     
    enum_<NeighborList::storageMode>("storageMode")
    .value("half", NeighborList::half)
    .value("full", NeighborList::full)
    ;
    }

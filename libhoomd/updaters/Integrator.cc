/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

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
// Maintainer: joaander

/*! \file Integrator.cc
    \brief Defines the Integrator base class
*/


#include "Integrator.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#ifdef ENABLE_CUDA
#include "Integrator.cuh"
#endif

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
*/
Integrator::Integrator(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT) : Updater(sysdef), m_deltaT(deltaT)
    {
    if (m_deltaT <= 0.0)
        m_exec_conf->msg->warning() << "integrate.*: A timestep of less than 0.0 was specified" << endl;
    }

Integrator::~Integrator()
    {
    #ifdef ENABLE_MPI
    // disconnect
    if (m_request_flags_connection.connected())
        m_request_flags_connection.disconnect();
    if (m_callback_connection.connected())
        m_callback_connection.disconnect();
    if (m_comm_callback_connection.connected())
        m_comm_callback_connection.disconnect();
    #endif
    }

/*! \param fc ForceCompute to add
*/
void Integrator::addForceCompute(boost::shared_ptr<ForceCompute> fc)
    {
    assert(fc);
    m_forces.push_back(fc);
    fc->setDeltaT(m_deltaT);
    }

/*! \param fc ForceConstraint to add
*/
void Integrator::addForceConstraint(boost::shared_ptr<ForceConstraint> fc)
    {
    assert(fc);
    m_constraint_forces.push_back(fc);
    fc->setDeltaT(m_deltaT);
    }

/*! Call removeForceComputes() to completely wipe out the list of force computes
    that the integrator uses to sum forces.
*/
void Integrator::removeForceComputes()
    {
    m_forces.clear();
    m_constraint_forces.clear();
    }

/*! \param deltaT New time step to set
*/
void Integrator::setDeltaT(Scalar deltaT)
    {
    if (m_deltaT <= 0.0)
        m_exec_conf->msg->warning() << "integrate.*: A timestep of less than 0.0 was specified" << endl;

    for (unsigned int i=0; i < m_forces.size(); i++)
        m_forces[i]->setDeltaT(deltaT);

    for (unsigned int i=0; i < m_constraint_forces.size(); i++)
        m_constraint_forces[i]->setDeltaT(deltaT);

     m_deltaT = deltaT;
    }

/*! \return the timestep deltaT
*/
Scalar Integrator::getDeltaT()
    {
    return m_deltaT;
    }

/*! Loops over all constraint forces in the Integrator and sums up the number of DOF removed
*/
unsigned int Integrator::getNDOFRemoved()
    {
    // start counting at 0
    unsigned int n = 0;

    // loop through all constraint forces
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_compute;
    for (force_compute = m_constraint_forces.begin(); force_compute != m_constraint_forces.end(); ++force_compute)
        n += (*force_compute)->getNDOFRemoved();

    return n;
    }

/*! The base class Integrator provides a few of the common logged quantities. This is the most convenient and
    sensible place to put it because most of the common quantities are computed by the various integrators.
    That, and there must be an integrator in any sensible simulation.ComputeThermo handles the computation of
    thermodynamic quantities.

    Derived integrators may also want to add additional quantities. They can do this in
    getProvidedLogQuantities() by calling Integrator::getProvidedLogQuantities() and adding their own custom
    provided quantities before returning.

    Integrator provides:
        - volume
        - box lengths lx, ly, lz
        - tilt factors xy, xz, yz
        - momentum
        - particle number N

    See Logger for more information on what this is about.
*/
std::vector< std::string > Integrator::getProvidedLogQuantities()
    {
    vector<string> result;
    result.push_back("volume");
    result.push_back("lx");
    result.push_back("ly");
    result.push_back("lz");
    result.push_back("xy");
    result.push_back("xz");
    result.push_back("yz");
    result.push_back("momentum");
    result.push_back("N");
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation

    The Integrator base class will provide a number of quantities (see getProvidedLogQuantities()). Derived
    classes that calculate any of these on their own can (and should) return their calculated values. To do so
    an overridden getLogValue() should have the following logic:
    \code
    if (quantitiy == "my_calculated_quantitiy1")
        return my_calculated_quantity1;
    else if (quantitiy == "my_calculated_quantitiy2")
        return my_calculated_quantity2;
    else return Integrator::getLogValue(quantity, timestep);
    \endcode
    In this way the "overriden" quantity is handled by the derived class and any other quantities are passed up
    to the base class to be handled there.

    See Logger for more information on what this is about.
*/
Scalar Integrator::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "volume")
        {
        BoxDim box = m_pdata->getGlobalBox();
        return box.getVolume(m_sysdef->getNDimensions()==2);
        }
    else if (quantity == "lx")
        {
        BoxDim box= m_pdata->getGlobalBox();
        Scalar3 L = box.getL();
        return L.x;
        }
    else if (quantity == "ly")
        {
        BoxDim box= m_pdata->getGlobalBox();
        Scalar3 L = box.getL();
        return L.y;
        }
    else if (quantity == "lz")
        {
        BoxDim box= m_pdata->getGlobalBox();
        Scalar3 L = box.getL();
        return L.z;
        }
    else if (quantity == "xy")
        {
        BoxDim box= m_pdata->getGlobalBox();
        Scalar xy = box.getTiltFactorXY();
        return xy;
        }
    else if (quantity == "xz")
        {
        BoxDim box= m_pdata->getGlobalBox();
        Scalar xz = box.getTiltFactorXZ();
        return xz;
        }
    else if (quantity == "yz")
        {
        BoxDim box= m_pdata->getGlobalBox();
        Scalar yz = box.getTiltFactorYZ();
        return yz;
        }
    else if (quantity == "momentum")
        {
        return computeTotalMomentum(timestep);
        }
    else if (quantity == "N")
        {
        return (Scalar) m_pdata->getNGlobal();
        }
    else
        {
        m_exec_conf->msg->error() << "integrate.*: " << quantity << " is not a valid log quantity for Integrator" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \param timestep Current timestep
    \post \c h_accel.data[i] is set based on the forces computed by the ForceComputes
*/
void Integrator::computeAccelerations(unsigned int timestep)
    {
#ifdef ENABLE_MPI
    if (m_comm)
        {
        // Move particles between domains. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if additionally ghosts are updated every timestep
        m_comm->forceMigrate();
        m_comm->communicate(timestep);
        }
#endif

    // compute the net forces
    computeNetForce(timestep);

    if (m_prof)
        {
        m_prof->push("Integrate");
        m_prof->push("Sum accel");
        }

    // now, get our own access to the arrays and calculate the accelerations
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);

    // now, add up the accelerations
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = h_net_force.data[j].x*minv;
        h_accel.data[j].y = h_net_force.data[j].y*minv;
        h_accel.data[j].z = h_net_force.data[j].z*minv;
        }

    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
    }

/*! \param timestep Current time step of the simulation

    computeTotalMomentum()  accesses the particle data on the CPU, loops through it and calculates the magnitude of the total
    system momentum
*/
Scalar Integrator::computeTotalMomentum(unsigned int timestep)
    {
    // grab access to the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

    // sum up the kinetic energy
    double p_tot_x = 0.0;
    double p_tot_y = 0.0;
    double p_tot_z = 0.0;
    for (unsigned int i=0; i < m_pdata->getN(); i++)
        {
        double mass = h_vel.data[i].w;
        p_tot_x += mass*(double)h_vel.data[i].x;
        p_tot_y += mass*(double)h_vel.data[i].y;
        p_tot_z += mass*(double)h_vel.data[i].z;
        }

    double p_tot = sqrt(p_tot_x * p_tot_x + p_tot_y * p_tot_y + p_tot_z * p_tot_z) / Scalar(m_pdata->getN());

    // done!
    return Scalar(p_tot);
    }

/*! \param timestep Current time step of the simulation
    \post All added force computes in \a m_forces are computed and totaled up in \a m_net_force and \a m_net_virial
    \note The summation step is performed <b>on the CPU</b> and will result in a lot of data traffic back and forth
          if the forces and/or integrater are on the GPU. Call computeNetForcesGPU() to sum the forces on the GPU
*/
void Integrator::computeNetForce(unsigned int timestep)
    {
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;
    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        (*force_compute)->compute(timestep);

    if (m_prof)
        {
        m_prof->push("Integrate");
        m_prof->push("Net force");
        }

    Scalar external_virial[6];
        {
        // access the net force and virial arrays
        const GPUArray<Scalar4>& net_force  = m_pdata->getNetForce();
        const GPUArray<Scalar>&  net_virial = m_pdata->getNetVirial();
        const GPUArray<Scalar4>& net_torque = m_pdata->getNetTorqueArray();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_net_torque(net_torque, access_location::host, access_mode::overwrite);

        // start by zeroing the net force and virial arrays
        memset((void *)h_net_force.data, 0, sizeof(Scalar4)*net_force.getNumElements());
        memset((void *)h_net_virial.data, 0, sizeof(Scalar)*net_virial.getNumElements());
        memset((void *)h_net_torque.data, 0, sizeof(Scalar4)*net_torque.getNumElements());

        for (unsigned int i = 0; i < 6; ++i)
           external_virial[i] = Scalar(0.0);

        // now, add up the net forces
        unsigned int nparticles = m_pdata->getN();
        unsigned int net_virial_pitch = net_virial.getPitch();
        assert(nparticles <= net_force.getNumElements());
        assert(6*nparticles <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
            {
            //phasing out ForceDataArrays
            //ForceDataArrays force_arrays = (*force_compute)->acquire();
            GPUArray<Scalar4>& h_force_array = (*force_compute)->getForceArray();
            GPUArray<Scalar>& h_virial_array = (*force_compute)->getVirialArray();
            GPUArray<Scalar4>& h_torque_array = (*force_compute)->getTorqueArray();

            ArrayHandle<Scalar4> h_force(h_force_array,access_location::host,access_mode::read);
            ArrayHandle<Scalar> h_virial(h_virial_array,access_location::host,access_mode::read);
            ArrayHandle<Scalar4> h_torque(h_torque_array,access_location::host,access_mode::read);

            unsigned int virial_pitch = h_virial_array.getPitch();
            for (unsigned int j = 0; j < nparticles; j++)
                {
                h_net_force.data[j].x += h_force.data[j].x;
                h_net_force.data[j].y += h_force.data[j].y;
                h_net_force.data[j].z += h_force.data[j].z;
                h_net_force.data[j].w += h_force.data[j].w;

                h_net_torque.data[j].x += h_torque.data[j].x;
                h_net_torque.data[j].y += h_torque.data[j].y;
                h_net_torque.data[j].z += h_torque.data[j].z;
                h_net_torque.data[j].w += h_torque.data[j].w;

                for (unsigned int k = 0; k < 6; k++)
                    {
                    h_net_virial.data[k*net_virial_pitch+j] += h_virial.data[k*virial_pitch+j];
                    }
                }

            for (unsigned int k = 0; k < 6; k++)
                external_virial[k] += (*force_compute)->getExternalVirial(k);
            }
        }

    for (unsigned int k = 0; k < 6; k++)
        m_pdata->setExternalVirial(k, external_virial[k]);

    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }

    // return early if there are no constraint forces
    if (m_constraint_forces.size() == 0)
        return;

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // communicate the net force
        m_comm->updateNetForce(timestep);
        }
    #endif

    // compute all the constraint forces next
    // constraint forces only apply a force, not a torque
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        (*force_constraint)->compute(timestep);

    if (m_prof)
        {
        m_prof->push("Integrate");
        m_prof->push("Net force");
        }

        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
        const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::overwrite);
        unsigned int net_virial_pitch = net_virial.getPitch();

        // now, add up the net forces
        unsigned int nparticles = m_pdata->getN();
        assert(nparticles <= net_force.getNumElements());
        assert(6*nparticles <= net_virial.getNumElements());
        for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
            {
            //phasing out ForceDataArrays
            //ForceDataArrays force_arrays = (*force_compute)->acquire();
            GPUArray<Scalar4>& h_force_array =(*force_constraint)->getForceArray();
            GPUArray<Scalar>& h_virial_array =(*force_constraint)->getVirialArray();
            ArrayHandle<Scalar4> h_force(h_force_array,access_location::host,access_mode::read);
            ArrayHandle<Scalar> h_virial(h_virial_array,access_location::host,access_mode::read);
            unsigned int virial_pitch = h_virial_array.getPitch();


            for (unsigned int j = 0; j < nparticles; j++)
                {
                h_net_force.data[j].x += h_force.data[j].x;
                h_net_force.data[j].y += h_force.data[j].y;
                h_net_force.data[j].z += h_force.data[j].z;
                h_net_force.data[j].w += h_force.data[j].w;
                for (unsigned int k = 0; k < 6; k++)
                    h_net_virial.data[k*net_virial_pitch+j] += h_virial.data[k*virial_pitch+j];
                }
            for (unsigned int k = 0; k < 6; k++)
                external_virial[k] += (*force_constraint)->getExternalVirial(k);
            }
        }

    for (unsigned int k = 0; k < 6; k++)
        m_pdata->setExternalVirial(k, external_virial[k]);

    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
    }

#ifdef ENABLE_CUDA
/*! \param timestep Current time step of the simulation
    \post All added frce computes in \a m_forces are computed and totaled up in \a m_net_force and \a m_net_virial
    \note The summation step is performed <b>on the GPU</b>.
*/
void Integrator::computeNetForceGPU(unsigned int timestep)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Cannot compute net force on the GPU if CUDA is disabled" << endl;
        throw runtime_error("Error computing accelerations");
        }

    // compute all the normal forces first
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;

    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        (*force_compute)->compute(timestep);

    if (m_prof)
        {
        m_prof->push(m_exec_conf, "Integrate");
        m_prof->push(m_exec_conf, "Net force");
        }

    Scalar external_virial[6];
        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force  = m_pdata->getNetForce();
        const GPUArray< Scalar4 >& net_torque = m_pdata->getNetTorqueArray();
        const GPUArray< Scalar >&  net_virial = m_pdata->getNetVirial();
        unsigned int net_virial_pitch = net_virial.getPitch();

        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar>  d_net_virial(net_virial, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_torque(net_torque, access_location::device, access_mode::overwrite);

        unsigned int nparticles = m_pdata->getN();
        assert(nparticles <= net_force.getNumElements());
        assert(nparticles*6 <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        // zero external virial
        for (unsigned int i = 0; i < 6; ++i)
            external_virial[i] = Scalar(0.0);

        // there is no need to zero out the initial net force and virial here, the first call to the addition kernel
        // will do that
        // ahh!, but we do need to zer out the net force and virial if there are 0 forces!
        if (m_forces.size() == 0)
            {
            // start by zeroing the net force and virial arrays
            cudaMemset(d_net_force.data, 0, sizeof(Scalar4)*net_force.getNumElements());
            cudaMemset(d_net_torque.data, 0, sizeof(Scalar4)*net_torque.getNumElements());
            cudaMemset(d_net_virial.data, 0, 6*sizeof(Scalar)*net_virial_pitch);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        // now, add up the accelerations
        // sum all the forces into the net force
        // perform the sum in groups of 6 to avoid kernel launch and memory access overheads
        for (unsigned int cur_force = 0; cur_force < m_forces.size(); cur_force += 6)
            {
            // grab the device pointers for the current set
            gpu_force_list force_list;

            const GPUArray<Scalar4>& d_force_array0 = m_forces[cur_force]->getForceArray();
            ArrayHandle<Scalar4> d_force0(d_force_array0,access_location::device,access_mode::read);
            const GPUArray<Scalar>& d_virial_array0 = m_forces[cur_force]->getVirialArray();
            ArrayHandle<Scalar> d_virial0(d_virial_array0,access_location::device,access_mode::read);
            const GPUArray<Scalar4>& d_torque_array0 = m_forces[cur_force]->getTorqueArray();
            ArrayHandle<Scalar4> d_torque0(d_torque_array0,access_location::device,access_mode::read);
            force_list.f0 = d_force0.data;
            force_list.v0 = d_virial0.data;
            force_list.vpitch0 = d_virial_array0.getPitch();
            force_list.t0 = d_torque0.data;

            if (cur_force+1 < m_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array1 = m_forces[cur_force+1]->getForceArray();
                ArrayHandle<Scalar4> d_force1(d_force_array1,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array1 = m_forces[cur_force+1]->getVirialArray();
                ArrayHandle<Scalar> d_virial1(d_virial_array1,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array1 = m_forces[cur_force+1]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque1(d_torque_array1,access_location::device,access_mode::read);
                force_list.f1 = d_force1.data;
                force_list.v1 = d_virial1.data;
                force_list.vpitch1 = d_virial_array1.getPitch();
                force_list.t1 = d_torque1.data;
                }
            if (cur_force+2 < m_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array2 = m_forces[cur_force+2]->getForceArray();
                ArrayHandle<Scalar4> d_force2(d_force_array2,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array2 = m_forces[cur_force+2]->getVirialArray();
                ArrayHandle<Scalar> d_virial2(d_virial_array2,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array2 = m_forces[cur_force+2]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque2(d_torque_array2,access_location::device,access_mode::read);
                force_list.f2 = d_force2.data;
                force_list.v2 = d_virial2.data;
                force_list.vpitch2 = d_virial_array2.getPitch();
                force_list.t2 = d_torque2.data;
                }
            if (cur_force+3 < m_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array3 = m_forces[cur_force+3]->getForceArray();
                ArrayHandle<Scalar4> d_force3(d_force_array3,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array3 = m_forces[cur_force+3]->getVirialArray();
                ArrayHandle<Scalar> d_virial3(d_virial_array3,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array3 = m_forces[cur_force+3]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque3(d_torque_array3,access_location::device,access_mode::read);
                force_list.f3 = d_force3.data;
                force_list.v3 = d_virial3.data;
                force_list.vpitch3 = d_virial_array3.getPitch();
                force_list.t3 = d_torque3.data;
                }
            if (cur_force+4 < m_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array4 = m_forces[cur_force+4]->getForceArray();
                ArrayHandle<Scalar4> d_force4(d_force_array4,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array4 = m_forces[cur_force+4]->getVirialArray();
                ArrayHandle<Scalar> d_virial4(d_virial_array4,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array4 = m_forces[cur_force+4]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque4(d_torque_array4,access_location::device,access_mode::read);
                force_list.f4 = d_force4.data;
                force_list.v4 = d_virial4.data;
                force_list.vpitch4 = d_virial_array4.getPitch();
                force_list.t4 = d_torque4.data;
                }
            if (cur_force+5 < m_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array5 = m_forces[cur_force+5]->getForceArray();
                ArrayHandle<Scalar4> d_force5(d_force_array5,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array5 = m_forces[cur_force+5]->getVirialArray();
                ArrayHandle<Scalar> d_virial5(d_virial_array5,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array5 = m_forces[cur_force+5]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque5(d_torque_array5,access_location::device,access_mode::read);
                force_list.f5 = d_force5.data;
                force_list.v5 = d_virial5.data;
                force_list.vpitch5 = d_virial_array5.getPitch();
                force_list.t5 = d_torque5.data;
                }

            // clear on the first iteration only
            bool clear = (cur_force == 0);

            // access flags
            PDataFlags flags = this->m_pdata->getFlags();

            gpu_integrator_sum_net_force(d_net_force.data,
                                         d_net_virial.data,
                                         net_virial_pitch,
                                         d_net_torque.data,
                                         force_list,
                                         nparticles,
                                         clear,
                                         flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    // add up external virials
    for (unsigned int cur_force = 0; cur_force < m_forces.size(); cur_force ++)
        for (unsigned int k = 0; k < 6; k++)
            external_virial[k] += m_forces[cur_force]->getExternalVirial(k);

    for (unsigned int k = 0; k < 6; k++)
        m_pdata->setExternalVirial(k, external_virial[k]);

    if (m_prof)
        {
        m_prof->pop(m_exec_conf);
        m_prof->pop(m_exec_conf);
        }

    // return early if there are no constraint forces
    if (m_constraint_forces.size() == 0)
        return;

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // communicate the net force
        m_comm->updateNetForce(timestep);
        }
    #endif

    // compute all the constraint forces next
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        (*force_constraint)->compute(timestep);

    if (m_prof)
        {
        m_prof->push("Integrate");
        m_prof->push(m_exec_conf, "Net force");
        }

        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
        const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
        const GPUArray< Scalar4 >& net_torque = m_pdata->getNetTorqueArray();
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_torque(net_torque, access_location::device, access_mode::overwrite);

        unsigned int nparticles = m_pdata->getN();
        assert(nparticles <= net_force.getNumElements());
        assert(6*nparticles <= net_virial.getNumElements());
        assert(nparticles <= net_torque.getNumElements());

        // now, add up the accelerations
        // sum all the forces into the net force
        // perform the sum in groups of 6 to avoid kernel launch and memory access overheads
        for (unsigned int cur_force = 0; cur_force < m_constraint_forces.size(); cur_force += 6)
            {
            // grab the device pointers for the current set
            gpu_force_list force_list;
            const GPUArray<Scalar4>& d_force_array0 = m_constraint_forces[cur_force]->getForceArray();
            ArrayHandle<Scalar4> d_force0(d_force_array0,access_location::device,access_mode::read);
            const GPUArray<Scalar>& d_virial_array0 = m_constraint_forces[cur_force]->getVirialArray();
            ArrayHandle<Scalar> d_virial0(d_virial_array0,access_location::device,access_mode::read);
            const GPUArray<Scalar4>& d_torque_array0 = m_constraint_forces[cur_force]->getTorqueArray();
            ArrayHandle<Scalar4> d_torque0(d_torque_array0,access_location::device,access_mode::read);
            force_list.f0 = d_force0.data;
            force_list.t0=d_torque0.data;
            force_list.v0 = d_virial0.data;
            force_list.vpitch0 = d_virial_array0.getPitch();

            if (cur_force+1 < m_constraint_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array1 = m_constraint_forces[cur_force+1]->getForceArray();
                ArrayHandle<Scalar4> d_force1(d_force_array1,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array1 = m_constraint_forces[cur_force+1]->getVirialArray();
                ArrayHandle<Scalar> d_virial1(d_virial_array1,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array1 = m_constraint_forces[cur_force + 1]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque1(d_torque_array1,access_location::device,access_mode::read);
                force_list.f1 = d_force1.data;
                force_list.t1=d_torque1.data;
                force_list.v1 = d_virial1.data;
                force_list.vpitch1 = d_virial_array1.getPitch();
                }
            if (cur_force+2 < m_constraint_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array2 = m_constraint_forces[cur_force+2]->getForceArray();
                ArrayHandle<Scalar4> d_force2(d_force_array2,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array2 = m_constraint_forces[cur_force+2]->getVirialArray();
                ArrayHandle<Scalar> d_virial2(d_virial_array2,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array2 = m_constraint_forces[cur_force + 2]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque2(d_torque_array2,access_location::device,access_mode::read);
                force_list.f2 = d_force2.data;
                force_list.t2=d_torque2.data;
                force_list.v2 = d_virial2.data;
                force_list.vpitch2 = d_virial_array2.getPitch();
                }
            if (cur_force+3 < m_constraint_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array3 = m_constraint_forces[cur_force+3]->getForceArray();
                ArrayHandle<Scalar4> d_force3(d_force_array3,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array3 = m_constraint_forces[cur_force+3]->getVirialArray();
                ArrayHandle<Scalar> d_virial3(d_virial_array3,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array3 = m_constraint_forces[cur_force + 3]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque3(d_torque_array3,access_location::device,access_mode::read);
                force_list.f3 = d_force3.data;
                force_list.t3=d_torque3.data;
                force_list.v3 = d_virial3.data;
                force_list.vpitch3 = d_virial_array3.getPitch();
                }
            if (cur_force+4 < m_constraint_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array4 = m_constraint_forces[cur_force+4]->getForceArray();
                ArrayHandle<Scalar4> d_force4(d_force_array4,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array4 = m_constraint_forces[cur_force+4]->getVirialArray();
                ArrayHandle<Scalar> d_virial4(d_virial_array4,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array4 = m_constraint_forces[cur_force + 4]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque4(d_torque_array4,access_location::device,access_mode::read);
                force_list.f4 = d_force4.data;
                force_list.t4=d_torque4.data;
                force_list.v4 = d_virial4.data;
                force_list.vpitch4 = d_virial_array4.getPitch();
                }
            if (cur_force+5 < m_constraint_forces.size())
                {
                const GPUArray<Scalar4>& d_force_array5 = m_constraint_forces[cur_force+5]->getForceArray();
                ArrayHandle<Scalar4> d_force5(d_force_array5,access_location::device,access_mode::read);
                const GPUArray<Scalar>& d_virial_array5 = m_constraint_forces[cur_force+5]->getVirialArray();
                ArrayHandle<Scalar> d_virial5(d_virial_array5,access_location::device,access_mode::read);
                const GPUArray<Scalar4>& d_torque_array5 = m_constraint_forces[cur_force + 5]->getTorqueArray();
                ArrayHandle<Scalar4> d_torque5(d_torque_array5,access_location::device,access_mode::read);
                force_list.f5 = d_force5.data;
                force_list.t5=d_torque5.data;
                force_list.v5 = d_virial5.data;
                force_list.vpitch5 = d_virial_array5.getPitch();
                }

            // clear only on the first iteration AND if there are zero forces
            bool clear = (cur_force == 0) && (m_forces.size() == 0);

            // access flags
            PDataFlags flags = this->m_pdata->getFlags();

            gpu_integrator_sum_net_force(d_net_force.data,
                                         d_net_virial.data,
                                         net_virial.getPitch(),
                                         d_net_torque.data,
                                         force_list,
                                         nparticles,
                                         clear,
                                         flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial]);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    // add up external virials
    for (unsigned int cur_force = 0; cur_force < m_constraint_forces.size(); cur_force ++)
        for (unsigned int k = 0; k < 6; k++)
            external_virial[k] += m_constraint_forces[cur_force]->getExternalVirial(k);

    for (unsigned int k = 0; k < 6; k++)
        m_pdata->setExternalVirial(k, external_virial[k]);


    if (m_prof)
        {
        m_prof->pop(m_exec_conf);
        m_prof->pop(m_exec_conf);
        }
    }
#endif

/*! The base class integrator actually does nothing in update()
    \param timestep Current time step of the simulation
*/
void Integrator::update(unsigned int timestep)
    {
    }

/*! prepRun() is to be called at the very beginning of each run, before any analyzers are called, but after the full
    simulation is defined. It allows the integrator to perform any one-off setup tasks and update net_force and
    net_virial, if needed.

    Specifically, updated net_force and net_virial in this call is a must for logged quantities to properly carry
    over in restarted jobs.

    The base class does nothing, it is up to derived classes to implement the correct behavior.
*/
void Integrator::prepRun(unsigned int timestep)
    {
    }

#ifdef ENABLE_MPI
/*! \param tstep Time step for which to determine the flags

    The flags needed are determiend by peeking to \a tstep and then using bitwise or
    to combine the flags from all ForceComputes
*/
CommFlags Integrator::determineFlags(unsigned int timestep)
    {
    CommFlags flags(0);

    // query all forces
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;
    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        flags |= (*force_compute)->getRequestedCommFlags(timestep);

    // query all constraints
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        flags |= (*force_constraint)->getRequestedCommFlags(timestep);

    return flags;
    }


void Integrator::setCommunicator(boost::shared_ptr<Communicator> comm)
    {
    // call base class method
    Updater::setCommunicator(comm);

    // connect to ghost communication flags request
    if (! m_request_flags_connection.connected() && m_comm)
        m_request_flags_connection = m_comm->addCommFlagsRequest(boost::bind(&Integrator::determineFlags, this, _1));

    if (! m_callback_connection.connected() && m_comm)
        m_callback_connection = comm->addComputeCallback(bind(&Integrator::computeCallback, this, _1));

    if (! m_comm_callback_connection.connected() && m_comm)
        m_comm_callback_connection = comm->addCommunicationCallback(bind(&Integrator::ghostCommunicationCallback, this, _1));
    }

void Integrator::computeCallback(unsigned int timestep)
    {
    // pre-compute all active forces
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;

    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        (*force_compute)->preCompute(timestep);
    }

void Integrator::ghostCommunicationCallback(const GPUArray<unsigned int>& plans)
    {
    // identify additonal ghost particles
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;

    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        (*force_compute)->addGhostParticles(plans);

    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        (*force_constraint)->addGhostParticles(plans);
    }
#endif

bool Integrator::getAnisotropic()
    {
    bool aniso = false;
    // pre-compute all active forces
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;

    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        aniso |= (*force_compute)->isAnisotropic();

    // pre-compute all active constraint forces
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        aniso |= (*force_constraint)->isAnisotropic();

    return aniso;
    }

void export_Integrator()
    {
    class_<Integrator, boost::shared_ptr<Integrator>, bases<Updater>, boost::noncopyable>
    ("Integrator", init< boost::shared_ptr<SystemDefinition>, Scalar >())
    .def("addForceCompute", &Integrator::addForceCompute)
    .def("addForceConstraint", &Integrator::addForceConstraint)
    .def("removeForceComputes", &Integrator::removeForceComputes)
    .def("setDeltaT", &Integrator::setDeltaT)
    .def("getNDOF", &Integrator::getNDOF)
    .def("getRotationalNDOF", &Integrator::getRotationalNDOF)
    ;
    }

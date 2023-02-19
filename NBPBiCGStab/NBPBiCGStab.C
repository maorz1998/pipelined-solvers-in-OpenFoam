/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "NBPBiCGStab.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(NBPBiCGStab, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<NBPBiCGStab>
        addPBiCGStabSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<NBPBiCGStab>
        addPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::NBPBiCGStab::NBPBiCGStab
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::NBPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    scalarField rM(nCells);
    scalar* __restrict__ rMPtr = rM.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalarField wM(nCells);
    scalar* __restrict__ wMPtr = wM.begin();

    scalarField tA(nCells);
    scalar* __restrict__ tAPtr = tA.begin();

    scalar alpha, beta;

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - yA);
    scalar* __restrict__ rAPtr = rA.begin();

    scalarField r0(rA);

    // --- Calculate normalisation factor
    const scalar normFactor = this->normFactor(psi, source, yA, pA);

    // --- Select and construct the preconditioner
    autoPtr<lduMatrix::preconditioner> preconPtr =
    lduMatrix::preconditioner::New
    (
        *this,
        controlDict_
    );
    preconPtr->precondition(rM, rA, cmpt);
    matrix_.Amul(wA, rM, interfaceBouCoeffs_, interfaces_, cmpt);
    preconPtr->precondition(wM, wA, cmpt);

    // initialize alpha & beta
    const scalar rArA = gSumSqr(rA, matrix().mesh().comm());
    scalar algha = rArA/gSumProd(rA, wA, matrix().mesh().comm());
    scalar beta = 0;
    scalar omega = 0;
    scalar r0rA = rArA;

    if (lduMatrix::debug >= 2)
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // Initialize parameters
        scalarField pM(nCells);
        scalar* __restrict__ pMPtr = pM.begin();

        scalarField sM(nCells);
        scalar* __restrict__ sMPtr = sM.begin();

        scalarField sA(nCells);
        scalar* __restrict__ sAPtr = sA.begin();

        scalarField zA(nCells);
        scalar* __restrict__ zAPtr = zA.begin();

        scalarField zM(nCells);
        scalar* __restrict__ zMPtr = zM.begin();

        scalarField vA(nCells);
        scalar* __restrict__ vAPtr = vA.begin();

        scalarField qA(nCells);
        scalar* __restrict__ qAPtr = qA.begin();

        scalarField qM(nCells);
        scalar* __restrict__ qMPtr = qM.begin();

        // --- Store initial residual
        const scalarField rA0(rA);

        // --- Solver iteration
        do
        {
            for (label cell=0; cell<nCells; cell++)
            {
                pMPtr[cell] = rMPtr[cell] + beta*(pMPtr[cell] - omega*sMPtr[cell]);
                sAPtr[cell] = wAPtr[cell] + beta*(sAPtr[cell] - omega*zAPtr[cell]);
                sMPtr[cell] = wMPtr[cell] + beta*(sMPtr[cell] - omega*zMPtr[cell]);
                zAPtr[cell] = tAPtr[cell] + beta*(zAPtr[cell] - omega*vAPtr[cell]);
                qAPtr[cell] = rAPtr[cell] - alpha*sAPtr[cell];
                qMPtr[cell] = rMPtr[cell] - alpha*sMPtr[cell];
                yAPtr[cell] = wAPtr[cell] - alpha*zAPtr[cell];
            }
            // check convergence
            solverPerf.finalResidual() =
                gSumMag(qA, matrix().mesh().comm())/normFactor;

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*pMPtr[cell];
                }

                solverPerf.nIterations()++;

                return solverPerf;
            }

            // --- overlap comm and comp
            // 1. no blocking reduction 
            int allredSend1[2];
            int allredRecv1[2];
            allredSend1[0] = sumProd(qA, yA);
            allredSend1[1] = sumSqr(yA);
            MPI_Request reqs;
            MPI_Iallreduce
            (
                &allredSend1,
                &allredRecv1,
                2,
                MPI_SCALAR,
                MPI_SUM,
                Pstream::msgType(),
                matrix().mesh().comm(),
                &reqs
            );
            // 2. SPMV
            preconPtr->precondition(zM, zA, cmpt);
            matrix_.Amul(vA, zM, interfaceBouCoeffs_, interfaces_, cmpt);
            // 3. MPI_wait
            MPI_Wait(&reqs, MPI_STATUS_IGNORE);
            
            omega = allredSend1[0] / allredSend1[1];
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pMPtr[cell] + omega*qMPtr[cell];
                rAPtr[cell] = qAPtr[cell] - omega*yAPtr[cell];
                rMPtr[cell] = qMPtr[cell] - omega*(qMPtr[cell] - alpha*zMPtr[cell]);
                wAPtr[cell] = yAPtr[cell] - omega*(tAPtr[cell] - alpha*vAPtr[cell]);
            }
            // check convergence
            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())/normFactor;

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*pMPtr[cell];
                }

                solverPerf.nIterations()++;

                return solverPerf;
            }

            // --- overlap comm and comp
            // 1. no blocking reduction 
            int allredSend2[4];
            int allredRecv2[4];
            allredSend2[0] = sumProd(r0, rA);
            allredSend2[1] = sumProd(r0, wA);
            allredSend2[2] = sumProd(r0, sA);
            allredSend2[3] = sumProd(r0, zA);
            MPI_Request reqs;
            MPI_Iallreduce
            (
                &allredSend2,
                &allredRecv2,
                4,
                MPI_SCALAR,
                MPI_SUM,
                Pstream::msgType(),
                matrix().mesh().comm(),
                &reqs
            );
            // 2. SPMV
            preconPtr->precondition(wM, wA, cmpt);
            matrix_.Amul(tA, wM, interfaceBouCoeffs_, interfaces_, cmpt);
            // 3. MPI_wait
            MPI_Wait(&reqs, MPI_STATUS_IGNORE);

            beta = (alpha/omega)*allredRecv2[0]/r0rA;
            alpha = allredRecv2[0]/(allredRecv2[1] + beta*allredRecv2[2] - beta*omega*allredRecv2[3]);
            r0rA = allredRecv2[0];

        } while
        (
            ++solverPerf.nIterations() < maxIter_ || solverPerf.nIterations() < minIter_
        );
    }

    return solverPerf;
}


// ************************************************************************* //

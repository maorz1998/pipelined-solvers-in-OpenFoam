/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2019 OpenFOAM Foundation
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

Class
    Foam::NBPBiCGStab

Description
    Preconditioned bi-conjugate gradient stabilized solver for asymmetric
    lduMatrices using a run-time selectable preconditioner.

    References:
    \verbatim
        Van der Vorst, H. A. (1992).
        Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG
        for the solution of nonsymmetric linear systems.
        SIAM Journal on scientific and Statistical Computing, 13(2), 631-644.

        Barrett, R., Berry, M. W., Chan, T. F., Demmel, J., Donato, J.,
        Dongarra, J., Eijkhout, V., Pozo, R., Romine, C. & Van der Vorst, H.
        (1994).
        Templates for the solution of linear systems:
        building blocks for iterative methods
        (Vol. 43). Siam.
    \endverbatim

SourceFiles
    NBPBiCGStab.C

\*---------------------------------------------------------------------------*/

#ifndef NBPBiCGStab_H
#define NBPBiCGStab_H

#include "lduMatrix.H"
#include "PstreamGlobals.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class NBPBiCGStab Declaration
\*---------------------------------------------------------------------------*/

class NBPBiCGStab
:
    public lduMatrix::solver
{

public:

    //- Runtime type information
    TypeName("NBPBiCGStab");


    // Constructors

        //- Construct from matrix components and solver data stream
        NBPBiCGStab
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        NBPBiCGStab(const NBPBiCGStab&) = delete;


    //- Destructor
    virtual ~NBPBiCGStab()
    {}


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const NBPBiCGStab&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

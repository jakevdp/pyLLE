include Make.inc

##########################################################################
# The rest should not need to be changed


#Define Object dependencies
MATVEC_OBJECTS = MatVec.o MatSym.o MatTri.o MatVecDecomp.o
MATVEC_TEST_OBJECTS = MatVec_test.o $(MATVEC_BASE)

#Define all executables
#all :  DimReduce areig_test MatVec_test
all: test lib

lib: libMatVec.so

test: MatVec_test

#Dependencies and compilation of each executable

MatVec_test: $(MATVEC_TEST_OBJECTS) libMatVec.so
	$(GCC) -fPIC $^ -o $@ $(GCC_LFLAGS) -lMatVec -L.

libMatVec.so: $(MATVEC_OBJECTS)
	$(GCC) -fPIC $^ -pthread -shared -o $@ $(GCC_LFLAGS) $(LFLAGS)

#object compilation
.cpp.o:
	$(GCC) -fPIC -c $< -o $@ $(GCC_CFLAGS)

.f.o:
	$(F77) -c $< -o $@ $(F77_CFLAGS)

clean :
	rm -f *.o
	rm -f *~ .*~

depend:
	makedepend *.cpp $(GCC_INCLUDE)

# below here are the automatically generated dependencies
# DO NOT DELETE
MatSym.o: MatSym.h MatVecExcept.h MatVecBase.h blas.h MatVec.h
MatTri.o: MatTri.h MatVecExcept.h MatVecBase.h blas.h MatVec.h
MatTri_test.o: MatTri.h MatVecExcept.h MatVecBase.h blas.h MatVec.h
MatVec.o: MatVec.h MatVecExcept.h MatVecBase.h blas.h
MatVecDecomp.o: MatVec.h MatVecExcept.h MatVecBase.h blas.h lapack.h arpack.h
MatVecDecomp.o: MatVecDecomp.h MatSym.h
MatVec_test.o: MatVec.h MatVecExcept.h MatVecBase.h blas.h MatSym.h MatTri.h
MatVec_test.o: MatVecDecomp.h /usr/include/time.h /usr/include/features.h
MatVec_test.o: /usr/include/sys/cdefs.h /usr/include/bits/wordsize.h
MatVec_test.o: /usr/include/gnu/stubs.h /usr/include/gnu/stubs-64.h
MatVec_test.o: /usr/include/bits/time.h /usr/include/bits/types.h
MatVec_test.o: /usr/include/bits/typesizes.h /usr/include/math.h
MatVec_test.o: /usr/include/bits/huge_val.h /usr/include/bits/mathdef.h
MatVec_test.o: /usr/include/bits/mathcalls.h /usr/include/stdlib.h
MatVec_test.o: /usr/include/sys/types.h /usr/include/endian.h
MatVec_test.o: /usr/include/bits/endian.h /usr/include/sys/select.h
MatVec_test.o: /usr/include/bits/select.h /usr/include/bits/sigset.h
MatVec_test.o: /usr/include/sys/sysmacros.h /usr/include/bits/pthreadtypes.h
MatVec_test.o: /usr/include/alloca.h

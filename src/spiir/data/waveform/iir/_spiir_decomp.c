#include <Python.h>

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include <lal/LALDatatypes.h>
#include <lal/LALStdlib.h>
#include <lal/Units.h>
#include <lal/LALInspiral.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <numpy/arrayobject.h>


/* static functions used by the python wrappers */

/* doc string for help() */
const char SPIIRDocstring[] =
  "This module wraps SPIIR coefficient generation from a CBC waveform template "
  "related to them.\n";

/* FIXME someday do better error handling? */
/* static PyObject* SPAWaveformError;      */

static PyObject *PyIIR(PyObject *self, PyObject *args)
{
	PyObject *amp, *phase;
	PyObject *amp_array, *phase_array;
	double eps, alpha, beta, padding;
	REAL8Vector amp_real8, phase_real8;
	COMPLEX16Vector *a1 =NULL;
	COMPLEX16Vector *b0 = NULL;
	INT4Vector *delay =NULL;
	PyObject *a1_pyob, *b0_pyob, *delay_pyob;
	npy_intp *amp_arraydims = NULL;
	npy_intp *phase_arraydims = NULL;
	npy_intp a1_length[] = {0};
	npy_intp b0_length[] = {0};
	npy_intp delay_length[] = {0};
	PyObject *out;

	if (!PyArg_ParseTuple(args, "OOdddd", &amp, &phase, &eps, &alpha, &beta, &padding)) return NULL;
	amp_array = PyArray_FROM_OTF(amp, NPY_DOUBLE, NPY_IN_ARRAY);
	phase_array = PyArray_FROM_OTF(phase, NPY_DOUBLE, NPY_IN_ARRAY);

	amp_arraydims = PyArray_DIMS(amp_array);
	amp_real8.length = amp_arraydims[0];
	amp_real8.data = PyArray_DATA(amp_array);
	phase_arraydims = PyArray_DIMS(phase_array);
	phase_real8.length = phase_arraydims[0];
	phase_real8.data = PyArray_DATA(phase_array);

	XLALInspiralGenerateIIRSet(&amp_real8, &phase_real8, eps, alpha, beta, padding, &a1, &b0, &delay);
	a1_length[0] = a1->length;
	b0_length[0] = b0->length;
	delay_length[0] = delay->length;
	a1_pyob = PyArray_SimpleNewFromData(1, a1_length, NPY_CDOUBLE, (void *) a1->data);
	b0_pyob = PyArray_SimpleNewFromData(1, b0_length, NPY_CDOUBLE, (void *) b0->data);
	delay_pyob = PyArray_SimpleNewFromData(1, delay_length, NPY_INT, (void *) delay->data);
	out = Py_BuildValue("OOO", a1_pyob, b0_pyob, delay_pyob);
	Py_DECREF(amp_array);
	Py_DECREF(phase_array);
	return out;
}

static PyObject *PyIIRResponse(PyObject *self, PyObject *args)
{
	PyObject *a1, *b0, *delay;
	PyObject *a1_array, *b0_array, *delay_array;
	int N;
	COMPLEX16Vector a1_complex16, b0_complex16;
	INT4Vector delay_int4;
	COMPLEX16Vector *resp =NULL;
	PyObject *resp_pyob;
	npy_intp *a1_arraydims = NULL;
	npy_intp *b0_arraydims = NULL;
	npy_intp *delay_arraydims = NULL;
	npy_intp resp_length[] = {0};

	if (!PyArg_ParseTuple(args, "iOOO", &N, &a1, &b0, &delay)) return NULL;
	a1_array = PyArray_FROM_OTF(a1, NPY_CDOUBLE, NPY_IN_ARRAY);
	b0_array = PyArray_FROM_OTF(b0, NPY_CDOUBLE, NPY_IN_ARRAY);
	delay_array = PyArray_FROM_OTF(delay, NPY_INT, NPY_IN_ARRAY);
	a1_arraydims = PyArray_DIMS(a1_array);
	a1_complex16.length = a1_arraydims[0];
	a1_complex16.data = PyArray_DATA(a1_array);
	b0_arraydims = PyArray_DIMS(b0_array);
	b0_complex16.length = b0_arraydims[0];
	b0_complex16.data = PyArray_DATA(b0_array);
	delay_arraydims = PyArray_DIMS(delay_array);
	delay_int4.length = delay_arraydims[0];
	delay_int4.data = PyArray_DATA(delay_array);

	resp = XLALCreateCOMPLEX16Vector(N);

	XLALInspiralIIRSetResponse(&a1_complex16, &b0_complex16, &delay_int4, resp);
	resp_length[0] = resp->length;
	resp_pyob = PyArray_SimpleNew(1, resp_length, NPY_CDOUBLE);
	memcpy(PyArray_DATA(resp_pyob), resp->data, resp->length * sizeof(*resp->data));
		
	XLALDestroyCOMPLEX16Vector(resp);
	Py_DECREF(a1_array);
	Py_DECREF(b0_array);
	Py_DECREF(delay_array);

	return resp_pyob;
}

static PyObject *PyIIRInnerProduct(PyObject *self, PyObject *args)
{
	PyObject *a1, *b0, *delay, *psd;
	REAL8 ip = 0.0;
	PyObject *a1_array, *b0_array, *delay_array, *psd_array;
	COMPLEX16Vector a1_complex16, b0_complex16;
	INT4Vector delay_int4;
	REAL8Vector psd_real8;
	npy_intp *a1_arraydims = NULL;
	npy_intp *b0_arraydims = NULL;
	npy_intp *delay_arraydims = NULL;
	npy_intp *psd_arraydims = NULL;

	if (!PyArg_ParseTuple(args, "OOOO", &a1, &b0, &delay, &psd)) return NULL;
	a1_array = PyArray_FROM_OTF(a1, NPY_CDOUBLE, NPY_IN_ARRAY);
	b0_array = PyArray_FROM_OTF(b0, NPY_CDOUBLE, NPY_IN_ARRAY);
	delay_array = PyArray_FROM_OTF(delay, NPY_INT, NPY_IN_ARRAY);
	psd_array = PyArray_FROM_OTF(psd, NPY_DOUBLE, NPY_IN_ARRAY);
	a1_arraydims = PyArray_DIMS(a1_array);
	a1_complex16.length = a1_arraydims[0];
	a1_complex16.data = PyArray_DATA(a1_array);
	b0_arraydims = PyArray_DIMS(b0_array);
	b0_complex16.length = b0_arraydims[0];
	b0_complex16.data = PyArray_DATA(b0_array);
	delay_arraydims = PyArray_DIMS(delay_array);
	delay_int4.length = delay_arraydims[0];
	delay_int4.data = PyArray_DATA(delay_array);
	psd_arraydims = PyArray_DIMS(psd_array);
	psd_real8.length = psd_arraydims[0];
	psd_real8.data = PyArray_DATA(psd_array);
	XLALInspiralCalculateIIRSetInnerProduct(&a1_complex16, &b0_complex16, &delay_int4, &psd_real8, &ip);
	Py_DECREF(a1_array);
	Py_DECREF(b0_array);
	Py_DECREF(delay_array);
	Py_DECREF(psd_array);

	return Py_BuildValue("d", ip);
}

/* Structure defining the functions of this module and doc strings etc... */
static struct PyMethodDef methods[] = {
	{
		"iir", PyIIR, METH_VARARGS,
		"This function calculates the a set of single pole IIR filters corresponding to a complex\n"
		"time series\n\n"
		"iir(amplitude, phase, epsilion, alpha, beta, padding\n\n"
	},
	{
		"iirresponse", PyIIRResponse, METH_VARARGS,
		"This function produces a truncated Impulse response function for a given delay of IIR filters with a1, b0 and delays.\n\n"
		"iirresponse(length_of_impulse_response, a1_set, b0_set, delay_set\n\n"
	},
	{
		"iirinnerproduct", PyIIRInnerProduct, METH_VARARGS,
		"This function outputs the inner product of the sum of iir responses\n\n"
		"iirinnerproduct(a1_set, b0_set, delay_set, psd\n\n"
	},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef spiirModule = {
	PyModuleDef_HEAD_INIT,
	"_spiir_decomp",
	SPIIRDocstring,
	-1,
	methods
};

PyMODINIT_FUNC PyInit__spiir_decomp(void)
{
    PyObject *module = PyModule_Create(&spiirModule);
    assert(! PyErr_Occurred());

	import_array();
	if (PyErr_Occurred()) {
        return NULL;
    }

	return module;
}
/*****************************************************************************/
/* The remainder of this code defines the static functions that the python   */
/* functions will use to compute various quantities.  They are not exposed   */
/* outside of this file directly but can be called from python via           */
/* the documentation described when doing help() on this module.             */
/* A lot of this code is in lal in some form or must be moved to lal.  Here  */
/* the functin prototypes have been vastly simplified (use of native c types */
/* and double precision)                                                     */
/*****************************************************************************/


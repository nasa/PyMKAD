#include <Python.h>
#include "lcs.h"
#include "numpy/arrayobject.h"
#include <stdio.h>

// Compatible with Python 3.7. 

// Function: compute nLCS
static PyObject* compute(PyObject* self, PyObject* args)
	{

		PyArrayObject *input1,*input2;
		
		if (!PyArg_ParseTuple(args, "OO",&input1,&input2))  
			return NULL;
		
		
		unsigned short *list1 = (unsigned short *)malloc(input1->dimensions[0]*sizeof(unsigned short));
		unsigned short *list2 = (unsigned short *)malloc(input2->dimensions[0]*sizeof(unsigned short));
		
	
		if(input1->dimensions[1]!=input2->dimensions[1]){
			fprintf(stderr,"Error Dimensions missmatch\n %d!=%d\n",input1->dimensions[1],input2->dimensions[1]);
			return Py_BuildValue("f",-1.0);
		}

		LCS LCSObj;
		float d=0;		
		for (int i=0;i<input1->dimensions[1];i++){
			for (int j=0;j<input1->dimensions[0];j++){
				memcpy(list1+j,input1->data+(i+j*input1->dimensions[1])*sizeof(unsigned short),sizeof(unsigned short));
			}
			for (int j=0;j<input2->dimensions[0];j++){
				memcpy(list2+j,input2->data+(i+j*input2->dimensions[1])*sizeof(unsigned short),sizeof(unsigned short));
			}
			d += LCSObj.CalcDist(list1,input1->dimensions[0],list2,input2->dimensions[0]);
		}
	
		free(list1);
		free(list2);
		
		return Py_BuildValue("f",d/((float)input1->dimensions[1]));
	}

// Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "compute", compute, METH_VARARGS, "Computes nLCS similarity" },
    { NULL, NULL, 0, NULL }
};

// Module Definition struct
static struct PyModuleDef nlcs = {
    PyModuleDef_HEAD_INIT,
    "nlcs",
    "Normalized Longest Common Subsequence Calculation",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_nlcs(void)
{
    return PyModule_Create(&nlcs);
}

 #include <iostream>   
 #include <stdlib.h>   
 #include <stdio.h>   
 #include <math.h>
 #include "lcs.h"

 using namespace std;	
    
// enum direction {north, west, nw};
// Direction uses chars {1,2,3};
    

LCS::LCS(){}


float LCS::CalcDist(unsigned short *vec1,unsigned short vec1Size,unsigned short *vec2,unsigned short vec2Size){

		unsigned short m, n;  	 //lengths of the two strings	
		unsigned short **c;		 // table of LCS lengths   
	     	char **b;     // table of which optimal subprob solution   
	        unsigned short i, j;   
		unsigned short length;	 // length of LCS of prefixes	

		m = (unsigned short)vec1Size;      // length of X   
		n = (unsigned short)vec2Size;      // length of Y   
    
		// Use two tables, b and c, each with m+1 rows and n+1 columns.   
		// Initialize the c table to all 0.  The b table doesn't need to be   
		// initialized.   	
		c =(unsigned short**) calloc(m+1, sizeof(unsigned short *));   
		for (i = 0; i <= m; i++)	
		{   
			c[i] =(unsigned short*) calloc(n+1, sizeof(unsigned short));   
		}   
		
		b = (char**)calloc(m+1, sizeof(char *));   
		for (i = 0; i <= m; i++)	
		{   
			b[i] = (char*)calloc(n+1, sizeof(char));   
		}   
		
    
		// Now run through the main loop of the LCS-Length algorithm on p.353.   
		for (i = 1; i <= m; i++)	
		{   
			for (j = 1; j <= n; j++)   
			{   
				if(vec1[i-1]==vec2[j-1])
				{   
				// Extending the LCS of X[1..i-1] and Y[1..j-1] by one character.   
					c[i][j] = c[i-1][j-1] + 1;   
					b[i][j] = 3;	//NorthWest
				}	
				else if (c[i-1][j] >= c[i][j-1])   
				{	
				// Using LCS of X[1..i-1] and Y[1..j].   
					c[i][j] = c[i-1][j];   
					b[i][j] = 1;   //North
				}	
				else   
				{	
				// Using LCS of X[1..i] and Y[1..j-1].   
					c[i][j] = c[i][j-1];   
					b[i][j] = 2;   //West
				}	
			}   
		}	
    
	//The tables are all filled in.  Print out the LCS found.   
	//print_LCS also returns the length of the LCS found.	
	length = print_LCS(b, m, n);   
//	printf("\nlength = %d\n", length);   
	for (i = 0; i <= m; i++)	
	{   
		free(c[i]);
	}   
	for (i = 0; i <= m; i++)	
	{   
		free(b[i]);
	}   
    	free(c);
	free(b);
	return (float)(length)/sqrt((float)(m)*(float)(n));

}

// Print an LCS of X[1..i] and Y[1..j], assuming that the b table has	
// already been filled in.  Based on the Print-LCS procedure of p.355.   
// int print_LCS(enum direction **b, char *X, int i, int j)   
//int LCS::print_LCS(enum direction **b,int i, int j)   
unsigned short LCS::print_LCS(char **b,unsigned short i, unsigned short j)   
{   
	if (i == 0 || j == 0)	// is either string empty?   
		return 0;   
	if (b[i][j] == 3)   //NorthWest
	{	
	// We extended X[1..i-1] and Y[1..j-1] by one character, which is X[i].   
	// Print the LCS of X[1..i-1] and Y[1..j-1] and then print X[i].   
		unsigned short length = print_LCS(b,i-1, j-1);   
		return length+1;   
	}	
	else if (b[i][j] == 1) //North
	{  
		return print_LCS(b,i-1, j); // used LCS of X[1..i-1] and Y[1..j]   
	}
		else   
	{
		return print_LCS(b,i, j-1); // used LCS of X[1..i] and Y[1..j-1]   
	}
    
}   


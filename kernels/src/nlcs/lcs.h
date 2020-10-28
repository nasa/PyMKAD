 #include <iostream>   
 #include <stdlib.h>   
 #include <stdio.h>   

using namespace std;	

/*enum direction {north, west, nw};*/

class LCS
{
	private:
		unsigned short print_LCS(char **b, unsigned short i, unsigned short j);

	public:
		LCS(void);
		float CalcDist(unsigned short *vec1,unsigned short vec1Size,unsigned short *str2,unsigned short vec2Size);

};

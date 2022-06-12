#include <bits/stdc++.h>
using namespace std;

int main()
{
	int t;
	scanf("%d", &t);

	while (t > 0)
	{
		int n;
		scanf("%d", &n);
		
		if (n < 100) {
			cout << n%10 << "\n";
		} else {
			int a = n;
			int c = 10;
			while (a != 0) {
				int r = a%10;
				c = min(c, r);
				a = a/10;
			}
			cout << c << "\n";
		}

		t--;
	}
	return 0;
}
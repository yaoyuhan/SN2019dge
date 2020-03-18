BEGIN {RS="";FS=","}
        {rec[++i]=$0 
	 a=$1; sub("^@.*{","",a)     # a=Zou2017
	 b[i]=a", "i                 # b[1]=Zou2017, 1
	}
END   { n=asort(b)                   #sort b[] alphabetically
	for (k=1;k<=n;k++)  { 
          m=split(b[k],outb,",");    #outb(m)=3  (corresponding to Arcavi2017)
          ind=outb[m];
          print rec[ind+0] "\n"      #subtlety: coerce ind to behave as integer
        }
      }


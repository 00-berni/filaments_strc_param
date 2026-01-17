#! /bin/sh
#++++++++++++++++
#.IDENTIFICATION tofits.sh
#.LANGUAGE       Bourne Shell
#.AUTHOR         CDS Catalogues Service (F. Ochsenbein)
#.ENVIRONMENT    Unix
#.KEYWORDS       FITS
#.VERSION  1.0   04-Mar-1993
#.VERSION  1.1   02-Dec-1993: Allows file to be compressed...
#.PURPOSE        Generate the FITS file to stdout
#.COMMENTS       This script transforms the set of input files + headers
#			into standard FITS TABLE EXTENSION files
#----------------
#
PATH="/bin:/usr/bin:/usr/ucb"
pgm=`basename $0`
usage="Usage: $0 file [file...] "
if [ $#argv -lt 1 ]; then
    echo "To generate the FITS equivalent onto standard output:"  1>&2
    echo "$usage"  1>&2
    exit 1
fi
#
#  Generate the standard SIMPLE header for ascii table
#
if [ -f SIMPLE.fih ]; then
    awk '{print} END { while (NR%36 != 0) { print ""; NR++}}' SIMPLE.fih \
    | dd conv=block cbs=80 bs=2880
else
  today__=`date +%d/%m/%y`
  ( echo "SIMPLE  =                    T / Standard FITS format"; \
    echo "BITPIX  =                    8 / Character information"; \
    echo "NAXIS   =                    0 / No image data array present"; \
    echo "EXTEND  =                    T / There may be standard extensions"; \
    echo "BLOCKED =                    T / The file may be blocked";\
    echo "ORIGIN  = 'CDS     '           / Written at CDS, Strasbourg/France";\
    echo "DATE    = '$today__'           / Date FITS table written (dd/mm/yy)";\
    echo "END";\
  ) | awk '{print} END { while (NR%36 != 0) { print ""; NR++}}' \
  | dd conv=block cbs=80 bs=2880
fi
#
#   Append the tables. One table is made of  file.fih  and  file
#
while [ $#argv -gt 0 ]; do
    if [ -f $1.Z ]; then		# Compressed file exists
	pcat=zcat
    else
	if [ -f $1 ]; then
	    pcat=cat
	else
	    echo "****Missing file: $1 or $1.Z"  1>&2
	    continue
	fi
    fi
    if [ -f $1.fih  ]; then
    	awk '{print} END { while (NR%36 != 0) { print ""; NR++}}' $1.fih \
    	| dd conv=block cbs=80 obs=2880		# First, the FITS Header...
	lr=`awk '/^NAXIS1 /{print $3}' $1.fih`
	nr=`awk '/^NAXIS2 /{print $3}' $1.fih`
    	echo "... File $1 = $nr x $lr bytes "  1>&2
	$pcat $1 | dd conv=block cbs=$lr obs=2880
	nb=`expr $lr '*' $nr '%' 2880`		# Number of bytes in last block
	# echo "nb=$nb" 1>&2
	if [ $nb -ne 0 ]; then
	    nb=`expr 2880 - $nb`
	    echo " " | dd conv=block cbs=$nb obs=2880
	fi
    else	# Error
	echo "****Missing file: $1.fih"  1>&2
    fi
    shift
done
exit 0

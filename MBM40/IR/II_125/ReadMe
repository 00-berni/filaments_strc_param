II/125      IRAS catalogue of Point Sources, Version 2.0 (IPAC 1986)
================================================================================
IRAS Catalog of Point Sources, Version 2.0
     Joint IRAS Science W.G.
    <IPAC (1986)>
    =1988IRASP.C......0J
================================================================================
ADC_Keywords: Photometry, infrared ; Infrared sources ; Surveys
Mission_Name: IRAS

Description:
    This is a catalog of some 250,000 well-confirmed infrared point
    sources observed by the Infrared Astronomical Satellite, i.e., sources
    with angular extents less than approximately 0.5, 0.5, 1.0, and 2.0
    arcmin in the in-scan direction at 12, 25, 60, and 100 microns,
    respectively. Positions, flux densities, uncertainties, associations
    with known astronomical objects and various cautionary flags are given
    for each object. While two other complementary data sets - the Working
    Survey Data Base and a file of rejected sources - give information
    about point-like sources, the information available in the Point
    Source Catalog should satisfy almost all users. Away from confused
    regions of the sky, the survey is complete to about 0.4, 0.5, 0.6, and
    1.0 Jy at 12, 25, 60, and 100 microns. Typical position uncertainties
    are about 2 to 6 arcseconds in-scan and about 8 to 16 arcseconds
    cross-scan. The processing steps applied to detect and confirm point
    sources, and the positional and photometric error analyses are
    described in the IRAS Catalogs and Atlases Explanatory Supplement; the
    catalog format is described in Chapter X. The sources appear in order
    of increasing (1950.0) right ascension.

    The included script "tofits.sh" should generate the FITS version of the
    tables on Unix platforms.

References:
  IRAS Catalogs and Atlases Explanatory Supplement, 1988, ed. Beichman C.,
     Neugebauer G., Habing H.J., Clegg P.E., and Chester T.J.
     (Washington, DC: GPO), NASA RP-1190, vol 1

See also:
    II/126 : IRAS Serendipitous Survey Catalog
    II/156 : IRAS Faint Source Catalog, |b| > 10, Version 2.0
    II/274 : IRAS Point Source Reject Catalog 
    VII/91 : IRAS Asteroid and Comet Survey (1986)
    II/190 : IRAS Minor Planet Survey data base (Tedesco, 1992)
   III/197 : IRAS Low Resolution Spectra (LRS) (1987)
   III/170 : IRAS Point Source Identifications (MacConnell, 1993)
    VII/73 : IRAS Small Scale Structure Catalog
   VII/113 : Catalogued Galaxies and QSOs observed in IRAS Survey,
                  Version 2 (IPAC 1989)
    II/174 : IRAS 2Jy Redshift Survey Data File
  J/A+AS/100/473 : IRAS pointed observations data (Assendorp+, 1993)
  J/AJ/109/2318  : Extragalactic IRAS Sources (Condon+ 1995)
  J/A+AS/99/31   : Identification of C stars in IRAS (Guglielmo+ 1993)
  J/A+AS/113/51  : Catalogue of associations IRAS and S stars (Chen+, 1995)
  J/ApJ/411/188  : Classification of bright IRAS variables (Allen+ 1993)
  J/A+AS/80/149  : CO observations of IRAS Sources (Wouterloot+ 1989)
  J/A+AS/98/589  : H2O, OH, CH3OH and CO obs. of IRAS Sources (Wouterloot+ 1993)
  J/A+AS/93/121  : CO emission from a sample of IRAS sources (Nyman+ 1992)
  J/A+AS/90/327  : 1612MHz OH survey of IRAS point sources
                                                (te Lintel Hekkert+ 1991)
    II/184 : Catalog of Infrared observations, 3rd Edition (Gezari+ 1993)

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl    Records    Explanations
--------------------------------------------------------------------------------
ReadMe          80          .    This file
psc.txt         79       1091    CD-ROM Documentation
SIMPLE.fih      81         44    FITS header
main.dat       157     245889    IRAS Point Sources
main.fih        80        294         associated FITS header
assoc.dat       58     142228    Associations
assoc.fih       80         66         associated FITS header
meaning.txt     80        235    Details of association catalogues
tofits.sh       80         74    Unix shell to generate FITS (1)
--------------------------------------------------------------------------------
(1) This small script should run on any Unix platform. To generate the FITS
    version of the tables, execute
    tofits main > main.fits       (for the main table)
    tofits assoc  > assoc.fits    (for the association file)
--------------------------------------------------------------------------------

Byte-per-byte Description of file: main.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1- 11  A11    ---     IRAS     IRAS source name
  12- 13  I2     h       RAh      Hours RA, equinox 1950.0, epoch 1983.5
  14- 15  I2     min     RAm      Minutes RA, equinox 1950.0, epoch 1983.5
  16- 18  I3     ds      RAds     Seconds RA, equinox 1950.0, epoch 1983.5
      19  A1     ---     DE-      Sign Dec, equinox 1950.0, epoch 1983.5
  20- 21  I2     deg     DEd      Degrees Dec, equinox 1950.0, epoch 1983.5
  22- 23  I2     arcmin  DEm      Minutes Dec, equinox 1950.0, epoch 1983.5
  24- 25  I2     arcsec  DEs      Seconds Dec, equinox 1950.0, epoch 1983.5
  26- 28  I3     arcsec  Major    Uncertainty ellipse major axis
  29- 31  I3     arcsec  Minor    Uncertainty ellipse minor axis
  32- 34  I3     deg     PosAng   Uncertainty ellipse position angle (1)
  35- 36  I2     ---     NHcon    Number of times observed
  37- 45  E9.3   Jy      Fnu_12   Average non-color corrected flux density,
                                    12um (5)
  46- 54  E9.3   Jy      Fnu_25   Average non-color corrected flux density,
                                    25um (5)
  55- 63  E9.3   Jy      Fnu_60   Average non-color corrected flux density,
                                    60um (5)
  64- 72  E9.3   Jy      Fnu_100  Average non-color corrected flux density,
                                    100um (5)
      73  I1     ---   q_Fnu_12   [1,3] Flux density quality, 12um (3)
      74  I1     ---   q_Fnu_25   [1,3] Flux density quality, 25um (3)
      75  I1     ---   q_Fnu_60   [1,3] Flux density quality, 60um (3)
      76  I1     ---   q_Fnu_100  [1,3] Flux density quality, 100um (3)
  77- 78  I2     ---     NLRS     Number of significant LRS spectra (4)
  79- 80  A2     ---     LRSChar  Characterization of averaged LRS spectrum (4)
  81- 83  I3     %     e_Fnu_12   Percent relative flux den. uncertainty, 12um
  84- 86  I3     %     e_Fnu_25   Percent relative flux den. uncertainty, 25um
  87- 89  I3     %     e_Fnu_60   Percent relative flux den. uncertainty, 60um
  90- 92  I3     %     e_Fnu_100  Percent relative flux den. uncertainty, 100um
  93- 97  I5     ---     TSNR_12  10x minimum signal-to-noise ratio, 12um
  98-102  I5     ---     TSNR_25  10x minimum signal-to-noise ratio, 25um
 103-107  I5     ---     TSNR_60  10x minimum signal-to-noise ratio, 60um
 108-112  I5     ---     TSNR_100 10x minimum signal-to-noise ratio, 100um
     113  A1     ---     CC_12    Point source correlation coeff., 12um (8)
     114  A1     ---     CC_25    Point source correlation coeff., 25um (8)
     115  A1     ---     CC_60    Point source correlation coeff., 60um (8)
     116  A1     ---     CC_100   Point source correlation coeff., 100um (8)
 117-118  I2     %       Var      Percent likelihood of variability
     119  A1     ---     Disc     Discrepant fluxes flag, 1 per band,
                                    hex encoded (6)
     120  A1     ---     Confuse  Confusion flags, 1 per band, hex encoded (6)
     121  I1     ---     PNearH   Number of nearby hours-confirmed point sources
     122  I1     ---     PNearW   Number of nearby weeks-confirmed point sources
     123  I1     ---     SES1_12  Nearby seconds-confirmed small ext., 12um (7)
     124  I1     ---     SES1_25  Nearby seconds-confirmed small ext., 25um (7)
     125  I1     ---     SES1_60  Nearby seconds-confirmed small ext., 60um (7)
     126  I1     ---     SES1_100 Nearby seconds-confirmed small ext., 100um (7)
     127  I1     ---     SES2_12  Nearby weeks-confirmed small ext., 12um (7)
     128  I1     ---     SES2_25  Nearby weeks-confirmed small ext., 25um (7)
     129  I1     ---     SES2_60  Nearby weeks-confirmed small ext., 60um (7)
     130  I1     ---     SES2_100 Nearby weeks-confirmed small ext., 100um (7)
     131  A1     ---     HSDFlag  High source density bin flag, hex encoded (6)
     132  I1     ---     Cirr1    Number of nearby 100 micron only WSDB sources
     133  I1     ---     Cirr2    100 micron sky brightness ratio to flux den.
                                    (2)
 134-136  I3     MJy/sr  Cirr3    Total 100 micron sky surface brightness
 137-138  I2     ---     NID      Number of positional associations
     139  I1     ---     IDType   [1,4] Type of association (9)
 140-141  I2     ---     MHcon    ? Possible number of HCONs
 142-145  I4     10-3    FCor_12  ? Flux correction factor applied (5)
 146-149  I4     10-3    FCor_25  ? Flux correction factor applied (5)
 150-153  I4     10-3    FCor_60  ? Flux correction factor applied (5)
 154-157  I4     10-3    FCor_100 ? Flux correction factor applied (5)
--------------------------------------------------------------------------------

Note (1): Measured in degrees east of north between the major axis of
     the ellipse and the local equatorial meridian.

Note (2): The spatially filtered 100 micron sky brightness to flux
     density ratio of point source (see Explanatory Supplement)

Note (3): 3=high quality, 2=moderate quality, 1=upper limit

Note (4): Low Resolution Spectra, see Cat. III/197

Note (5): The flux densities listed in the catalog are average values
     computed for the effective wavelengths of the IRAS bands and an
     assumed input energy distribution F({nu}){prop.to}1/{nu} (i.e. a
     spectral index of -1). See details in the Explanatory Supplement
     or at the IRAS archive site at
     http://lambda.gsfc.nasa.gov/product/iras/colorcorr.cfm

Note (6): The hexadecimal encoding is the representation of a 4-bit
     number with values set to 1=12um, 2=25um, 4=60um, 8=100um. For
     instance, the value A=10=8+2 has flags set for 25 and 100um.

Note (7): Values of SES1 greater than 1 should caution the user that
     significant extended structure may exist in the region and that
     the source in question may be a point-source like piece of a
     complex field. SES2 values greater than 0 means that the point
     source flux measurement should be treated with caution as the
     source in question may, in fact, be extended; the flux quoted
     in the catalog of small extended sources (Cat. VII/73) may
     provide a better value for the source.

Note (8): from section V.C.4 of the Explanation:
    the point source correlation coefficient can have values in the range
    87-100%, coded as alphabetic characters with A=100, B=99 .. N=87

Note (9): Type of association with:
      1 = extragalactic catalog ;
      2 = stellar catalog ;
      3 = other catalogs ;
      4 = multiple types of catalogs.
--------------------------------------------------------------------------------


Byte-per-byte Description of file: assoc.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1- 11  A11    ---     IRAS     IRAS source name
  12- 17  I6     ---     RecNo    Main data table record number for IRAS source
  19- 20  I2     ---     CatNum   Catalog number (1)
  21- 35  A15    ---     Source   Source identification
  36- 40  A5     ---     Type     Source type or spectral class
  41- 43  I3     arcsec  Radius   Radius vector from IRAS source to association
  44- 46  I3     deg     Pos      Position angle (E of N), IRAS source to object
  47- 50  I4     ---     Field1   Object data field 1, catalog dependent
  51- 54  I4     ---     Field2   Object data field 2, catalog dependent
  55- 58  I4     ---     Field3   Object data field 3, catalog dependent
--------------------------------------------------------------------------------
Note (1): See Table X.B.4 in  the "IRAS Catalogs and Atlases Explanatory
    Supplement" also included in the "meaning.txt" file.
--------------------------------------------------------------------------------

Historical Notes:
    The catalogue is also available on the CD-ROM
    "Selected Astronomical Catalogues", Volume 1, 1992,
    Astronomical Data Center, NASA, directory photom/iraspsc
  * 10-Jun-2004: in file "assoc.dat", record#133500, a control-character
    (in Field 2 representing a spectral type) was replaced.
================================================================================
(End)                                                        [CDS]   29-Jan-1994

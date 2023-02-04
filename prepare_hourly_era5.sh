#!/bin/bash -l
#SBATCH --job-name=prepare_era5
#SBATCH --time=0-09:00:00
#SBATCH --mem=3gb
#SBATCH -o /home/oboucher/prepare_era5.%j

#--script merges reanalysis and 3-hr forecast to produce a hourly file of winds 
#--in doing this the longitude becomes [-180,180] instead of [0,360]
#--and xarray expects the same otherwise extraction is wrong !!!!

#--dirout
dirout=/projsu/cmip-work/oboucher/ERA5

#--Globe
lonmin=-180
lonmax=180
latmin=-90
latmax=90
ext=GLOBAL

#--year
yr=2022
yr2=$((yr+1))

rm -f ${dirout}/${var}.??????.ap1e5.GLOBAL_025.nc

#for mth in {01..12}
for mth in {10..10}
do
for var in u v r ta
do
echo $yr $mth
dirin=/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/${yr}
file=${var}.${yr}${mth}.ap1e5.GLOBAL_025.nc
cdo sellonlatbox,${lonmin},${lonmax},${latmin},${latmax} -sellevel,175,200,225,250 ${dirin}/${file} ${dirout}/${file}
done
done

#for mth in {01..01}
#do
#for var in u v r ta
#do
#echo $yr2 $mth
#dirin=/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL/${yr2}
#file=${var}.${yr2}${mth}.ap1e5.GLOBAL_025.nc
#cdo sellonlatbox,${lonmin},${lonmax},${latmin},${latmax} -sellevel,175,200,225,250 ${dirin}/${file} ${dirout}/${file}
#done
#done

#for var in u v r ta
#do
#cdo mergetime ${dirout}/${var}.${yr}??.ap1e5.GLOBAL_025.nc ${dirout}/${var}.${yr2}01.*.GLOBAL_025.nc ${dirout}/${var}.${yr}.${ext}.nc
#rm -f ${dirout}/${var}.${yr}??.*.GLOBAL_025.nc ${dirout}/${var}.${yr2}01.*.GLOBAL_025.nc
#done

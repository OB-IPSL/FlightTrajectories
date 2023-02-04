#!/bin/bash

yr=2018
yr=2019

for file in FLIGHTS/*${yr}.csv
#for file in FLIGHTS/FLIGHTS/asia_to_southeastasia_2019.csv

do

route=$(basename $file _${yr}'.csv')
echo $route

cat > optim_${route}_${yr}.job << END_JOB
#!/bin/bash -l
#SBATCH --job-name=optim_${route}_${yr}
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/oboucher/optim_${route}_${yr}.%j

cd /home/oboucher/AVION/OPTIM
python optim_iagos_only.py --yr=${yr} --route=${route} --level=-1

END_JOB

echo optim_${route}_${yr}.job
sbatch optim_${route}_${yr}.job
mv optim_${route}_${yr}.job JOBS

done

host=wup35031

ssh $host mkdir /tmp/tyam
dt=`ssh $host date "+%Y%m%d_%H%M%S"`
dir="/tmp/tyam/0190_jk_len2d_$dt"
ssh $host mkdir $dir
scp ../target/uberjar/mlp-0.1.0-SNAPSHOT-standalone.jar $host:$dir/run.jar
cat <<EOS | ssh $host "at 6:43"
mkdir $dir/result
cd $dir
java -cp run.jar mlp.core 5 5 1001 0.1 1 5 5 > result/__5__5__5__5.log
EOS

host=dn34460l

ssh $host mkdir /tmp/tyam
dt=`ssh $host date "+%Y%m%d_%H%M%S"`
dir="/tmp/tyam/0210_schemn_$dt"
ssh $host mkdir $dir
scp ../target/mlp-0.1.0-SNAPSHOT-standalone.jar $host:$dir/run.jar
cat <<EOS | ssh $host "at 13:06"
mkdir $dir/result
cd $dir
java -cp run.jar mlp.schem2 100001 0.1  1 5 5 > result/__1__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  2 5 5 > result/__2__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  3 5 5 > result/__3__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  4 5 5 > result/__4__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  5 5 5 > result/__5__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  6 5 5 > result/__6__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  7 5 5 > result/__7__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  8 5 5 > result/__8__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  9 5 5 > result/__9__5__5.log
java -cp run.jar mlp.schem2 100001 0.1 10 5 5 > result/_10__5__5.log
java -cp run.jar mlp.schem2 100001 0.1  1 3 5 > result/__1__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  2 3 5 > result/__2__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  3 3 5 > result/__3__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  4 3 5 > result/__4__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  5 3 5 > result/__5__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  6 3 5 > result/__6__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  7 3 5 > result/__7__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  8 3 5 > result/__8__3__5.log
java -cp run.jar mlp.schem2 100001 0.1  9 3 5 > result/__9__3__5.log
java -cp run.jar mlp.schem2 100001 0.1 10 3 5 > result/_10__3__5.log
EOS

#include<cstdio>
#include<cmath>
#include<cstdlib>
#include<vector>


//
// result_COPのディレクトリを作成
// 内部にdumpとresultを作成
// 

//----書き換える必要があるところ----//
#define SUBNUM 11 //被験者数
#define TASKNUM 5 			//タスク数(比較姿勢数)
#define TRIALNUM 5		//試行回数


#define CHNUM	15					//チャネル数
#define MAXDATA 29000				//最大計測データ数(最大タスク実行時間＊サンプリング)
#define MAXTEXT 256					//最大データ数
#define FALLTH	1.0					//転倒閾値(1N)
#define DEF_PI  3.14159265			//円周率

const char* taskName[] = {"NC","FB","D1","D2","DW"};//タスク名
const char* datafile = "241223";//データの位置


//構造体変数定義

//フォースプレートデータ
struct ForceplateData{
	double forcetime;
	int datanum;
	double forcex;
	double forcey;
	double forcez;
	double momentx;
	double momenty;
	double momentz;
	double momentdashx;
	double momentdashy;
	double axisx;
	double axisy;
	double torque;
	double cofx;
	double cofy;
	double cofz;
};

//評価指標データ
struct EvaluationIndex{
	double fallcheck;			//転倒判定
	double counterforce;		//床反力
	double holdtime;
	double displacementx;		//軌跡長x
	double displacementy;		//軌跡長y
	double displacementxy;		//軌跡長
	double totalDx;
	double totalDy;
	double totalDxy;
	double rmsx;				//実行値（標準偏差）x
	double rmsy;				//実行値（標準偏差）y
	double rmsxy;				//実行値
	double ampwidthx;			//振幅幅x
	double ampwidthy;			//振幅幅y
	double peripheryare;		//外周面積
	double rectanglearea;		//矩形面積
	double rmsarea;				//実行値面積
	double sdarea;				//標準偏差面積
};

//両足の重心位置
class CenterData{
private:
	int dataNum;
	EvaluationIndex indexData;
	double maxAxisX, maxAxisY, minAxisX, minAxisY;
	double counterForce;
	std::pair<double, double> sumOfAxis;
	std::vector<double> axisx, axisy, forcez;


	void setAmpWidthX(const double &ax)
	{
		if(ax < minAxisX){
			minAxisX = ax;
		} else if (ax > maxAxisX){
			maxAxisX = ax;
		}
	}

	void setAmpWidthY(const double &ay)
	{
		if(ay < minAxisY){
			minAxisY = ay;
		} else if(ay > maxAxisY){
			maxAxisY = ay;
		}
	}

	double meanAxisX(){
		return sumOfAxis.first / axisx.size();
	}

	double meanAxisY(){
		return sumOfAxis.second / axisy.size();
	}

	double ampWidthX(){
		return maxAxisX - minAxisX;
	}

	double ampWidthY(){
		return maxAxisY - minAxisY;
	}

	void calcSD(){
		indexData.rmsx = indexData.rmsy = indexData.rmsxy = 0.0;

		double meanX = meanAxisX(), meanY = meanAxisY();
		for(int i = 0; i < dataNum; i++){
			indexData.rmsx += pow(axisx[i] - meanX, 2);
			indexData.rmsy += pow(axisy[i] - meanY, 2);
			indexData.rmsxy += pow(axisx[i] - meanX, 2) + pow(axisy[i] - meanY, 2);
		}
		indexData.rmsx = sqrt(indexData.rmsx/dataNum);
		indexData.rmsy = sqrt(indexData.rmsy/dataNum);
		indexData.rmsxy = sqrt(indexData.rmsxy/dataNum);
	}

public:
	void clear(){
		axisx.clear();
		axisy.clear();
		forcez.clear();
		maxAxisX = maxAxisY = -1000;
		minAxisX = minAxisY = 1000;
		sumOfAxis.first = sumOfAxis.second = 0.0;
		counterForce = 0;
		dataNum = 0;
		indexData = EvaluationIndex();
	}

	void set(const ForceplateData &front, const ForceplateData &rear){
		dataNum++;

		//set time
		indexData.holdtime = front.forcetime;

		double sumOfForce = front.forcez + rear.forcez;

		double frontx = front.axisx + 150;
		double rearx = rear.axisx - 150;

		double ax = (frontx * front.forcez + rearx * rear.forcez) / sumOfForce;
		double ay = (front.axisy * front.forcez + rear.axisy * rear.forcez) / sumOfForce;

		//COP座標とFzを格納
		axisx.push_back(ax);
		axisy.push_back(ay);
		forcez.push_back(sumOfForce);



		//振幅を求める
		setAmpWidthX(ax);
		setAmpWidthY(ay);

		//座標の平均値を出すために全部足しておく
		sumOfAxis.first += ax;
		sumOfAxis.second += ay;

		counterForce += (double)sqrt((double)(front.forcex*front.forcex + front.forcey*front.forcey +front.forcez * front.forcez)); //分力の合計値
		counterForce += (double)sqrt((double)(rear.forcex*rear.forcex + rear.forcey*rear.forcey + rear.forcez * rear.forcez)); //分力の合計値
	}

	bool isFailure()
	{
		return (forcez.back() < forcez.front() * 0.5);//最新の値が初期値の半分以下なら転倒
	}

	EvaluationIndex getEvalationIndex()
	{
		indexData.fallcheck = (dataNum < MAXDATA) ? 0 : 1;
		indexData.counterforce = counterForce/axisy.size();
		double dx, dy;
		for(int i = 1; i < dataNum; i++){
			dx = axisx[i] - axisx[i-1];
			dy = axisy[i] - axisy[i-1];

			indexData.totalDx += fabs(dx);
			indexData.totalDy += fabs(dy);
			indexData.totalDxy += sqrt(pow(dx,2) + pow(dy,2));
		}
		indexData.displacementx = indexData.totalDx / dataNum;
		indexData.displacementy = indexData.totalDy / dataNum;
		indexData.displacementxy = indexData.totalDxy / dataNum;
		indexData.ampwidthx = ampWidthX();
		indexData.ampwidthy = ampWidthY();
		calcSD();
		indexData.rectanglearea = indexData.ampwidthx * indexData.ampwidthy;
		indexData.rmsarea = indexData.rmsxy * indexData.rmsxy * DEF_PI;
		indexData.sdarea = indexData.rmsx * indexData.rmsy * DEF_PI;
		return indexData;
	}
	void dump(int subNumber, int taskNumber, int triNumber)
	{
		char buf[64];
		FILE* fpw;
		sprintf(buf, "D:/User/kanai/Data/%s/result_COP/dump/sub%02d%s%02d.csv", datafile, subNumber + 1, taskName[taskNumber], triNumber + 1);
		if((fpw = fopen(buf, "w")) == NULL){
			printf("cannot create %s\n", buf);
			return;
		}
		fprintf(fpw, "ax,ay,Fz\n");
// 変更点　最後に外れ値が出るようになっていたのでそれを除いた
		for(int i = 0; i+1 < dataNum; i++){
			fprintf(fpw, "%lf,%lf,%lf\n",axisx[i], axisy[i], forcez[i]);
		}
		fclose(fpw);
	}
};

//読み込みファイルオープン（フォースプレート）
bool openFiles(FILE* &fp, char* readfileName, int subNumber, int taskNumber, int triNumber, int plate_id){
	sprintf(readfileName,"D:/User/kanai/Data/%s/sub%d/csv/fp/%s%04d%_f_%d.csv", datafile, subNumber + 1, taskName[taskNumber], triNumber + 1, plate_id + 1);	//読み込みファイル名設定
	if((fp = fopen(readfileName,"r")) == NULL){
		printf("%s open error\n",readfileName);
		return false;
	} else {
		printf("open %s\n", readfileName);
	}
	return true;
}
//ラベル削除
void removeHeader(FILE* &fp){
	char buf[1024];
	fscanf(fp,"%s%s%s%s%s%s%s%s%s",&buf,&buf,&buf,&buf,&buf,&buf,&buf,&buf,&buf);
}

//計測データ代入
void loadData(FILE* &fp, ForceplateData *dist){
	char buf[1024];
	fscanf(fp,"%s",buf);
	sscanf(buf,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &dist->forcetime,&dist->forcex,&dist->forcey,&dist->forcez,&dist->momentx,&dist->momenty,&dist->momentz,&dist->momentdashx,&dist->momentdashy,&dist->axisx,&dist->axisy,&dist->torque,&dist->cofx,&dist->cofy,&dist->cofz);	//文字から数値に変換
}

int main(void){

	CenterData centerData;//両足のCOP
	ForceplateData frontData;//前に出してる足のほうのデータ
	ForceplateData rearData;//後ろ側の足のデータ

	EvaluationIndex indexdata;		//評価指標データ

	char readfilename[MAXTEXT];				//読み込みファイル名
	char writefilename[MAXTEXT];			//書き込みファイル名(各結果)

	FILE *fpFront, *fpRear;					//読み込みファイル（フォースプレート）
	FILE *fpw;								//書き込みファイル

	int subnumber;						//被験者数
	int tasknumber;						//タスク数
	int trinumber;						//試行数

	for(subnumber = 0; subnumber<SUBNUM; subnumber++){

		//書き込みファイルオープン(最終結果)
		sprintf(writefilename,"D:/User/kanai/Data/%s/result_COP/result/sub%dresult.csv", datafile, subnumber + 1);		//書き込みファイル名設定
		if((fpw = fopen(writefilename,"w")) == NULL){
			printf("endresult open error. cannot open%s\n", writefilename);
			continue;
		}

		for(tasknumber = 0; tasknumber<TASKNUM; tasknumber++){

			//ラベル書き込み
			fprintf(fpw,"タスク番号,");
			fprintf(fpw,"試行数,");
			fprintf(fpw,"転倒有無,holdtime,床反力,");
			fprintf(fpw,"総軌跡長x,総軌跡長y,総軌跡長,");
			fprintf(fpw,"単位軌跡長x,単位軌跡長y,単位軌跡長,");
			fprintf(fpw,"SDx,SDy,実行値,");
			fprintf(fpw,"振幅幅x,振幅幅y,");
			fprintf(fpw,"矩形面積,実行値面積,標準偏差面積,");
			fprintf(fpw,"\n");

			for(trinumber = 0; trinumber<TRIALNUM; trinumber++){

				if(!openFiles(fpRear, readfilename, subnumber, tasknumber, trinumber, 0)){
					return -1;
				}
				if(!openFiles(fpFront, readfilename, subnumber, tasknumber, trinumber, 1)){
					return -1;
				}

				//ファイルのヘッダを削除
				removeHeader(fpRear);
				removeHeader(fpFront);

				centerData.clear();

				while(1){
					loadData(fpRear, &rearData);
					loadData(fpFront, &frontData);

					centerData.set(frontData, rearData);

					if(centerData.isFailure())	break;

					//ファイル端末判定
					if(feof(fpFront) != 0)							break;
				}

				/////////////////////////////////////////////
				//評価指標算出
				/////////////////////////////////////////////
				indexdata = centerData.getEvalationIndex();

				//check centerData
				centerData.dump(subnumber, tasknumber, trinumber);

				//結果記述
				printf("Sub.%d タスク%d 試行%dの解析終了！\n",subnumber+1,tasknumber+1,trinumber+1);
				fprintf(fpw,"%d,",tasknumber+1);
				fprintf(fpw,"%d,",trinumber+1);
				fprintf(fpw,"%lf,",indexdata.fallcheck);
				fprintf(fpw,"%lf,",indexdata.holdtime);
				fprintf(fpw,"%lf,",indexdata.counterforce);
				fprintf(fpw,"%lf,%lf,%lf,",indexdata.totalDx,indexdata.totalDy,indexdata.totalDxy);
				fprintf(fpw,"%lf,%lf,%lf,",indexdata.displacementx ,indexdata.displacementy,indexdata.displacementxy);
				fprintf(fpw,"%lf,%lf,%lf,",indexdata.rmsx ,indexdata.rmsy,indexdata.rmsxy);
				fprintf(fpw,"%lf,%lf,",indexdata.ampwidthx ,indexdata.ampwidthy);
				fprintf(fpw,"%lf,%lf,%lf,",indexdata.rectanglearea ,indexdata.rmsarea,indexdata.sdarea);
				fprintf(fpw,"\n");

				fclose(fpFront);
				fclose(fpRear);
			}
		}
		fclose(fpw);
	}
	return 0;
}
#include<cstdio>
#include<cmath>
#include<cstdlib>
#include<vector>


//
// result_COP�̃f�B���N�g�����쐬
// ������dump��result���쐬
// 

//----����������K�v������Ƃ���----//
#define SUBNUM 11 //�팱�Ґ�
#define TASKNUM 5 			//�^�X�N��(��r�p����)
#define TRIALNUM 5		//���s��


#define CHNUM	15					//�`���l����
#define MAXDATA 29000				//�ő�v���f�[�^��(�ő�^�X�N���s���ԁ��T���v�����O)
#define MAXTEXT 256					//�ő�f�[�^��
#define FALLTH	1.0					//�]�|臒l(1N)
#define DEF_PI  3.14159265			//�~����

const char* taskName[] = {"NC","FB","D1","D2","DW"};//�^�X�N��
const char* datafile = "241223";//�f�[�^�̈ʒu


//�\���̕ϐ���`

//�t�H�[�X�v���[�g�f�[�^
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

//�]���w�W�f�[�^
struct EvaluationIndex{
	double fallcheck;			//�]�|����
	double counterforce;		//������
	double holdtime;
	double displacementx;		//�O�Ւ�x
	double displacementy;		//�O�Ւ�y
	double displacementxy;		//�O�Ւ�
	double totalDx;
	double totalDy;
	double totalDxy;
	double rmsx;				//���s�l�i�W���΍��jx
	double rmsy;				//���s�l�i�W���΍��jy
	double rmsxy;				//���s�l
	double ampwidthx;			//�U����x
	double ampwidthy;			//�U����y
	double peripheryare;		//�O���ʐ�
	double rectanglearea;		//��`�ʐ�
	double rmsarea;				//���s�l�ʐ�
	double sdarea;				//�W���΍��ʐ�
};

//�����̏d�S�ʒu
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

		//COP���W��Fz���i�[
		axisx.push_back(ax);
		axisy.push_back(ay);
		forcez.push_back(sumOfForce);



		//�U�������߂�
		setAmpWidthX(ax);
		setAmpWidthY(ay);

		//���W�̕��ϒl���o�����߂ɑS�������Ă���
		sumOfAxis.first += ax;
		sumOfAxis.second += ay;

		counterForce += (double)sqrt((double)(front.forcex*front.forcex + front.forcey*front.forcey +front.forcez * front.forcez)); //���͂̍��v�l
		counterForce += (double)sqrt((double)(rear.forcex*rear.forcex + rear.forcey*rear.forcey + rear.forcez * rear.forcez)); //���͂̍��v�l
	}

	bool isFailure()
	{
		return (forcez.back() < forcez.front() * 0.5);//�ŐV�̒l�������l�̔����ȉ��Ȃ�]�|
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
// �ύX�_�@�Ō�ɊO��l���o��悤�ɂȂ��Ă����̂ł����������
		for(int i = 0; i+1 < dataNum; i++){
			fprintf(fpw, "%lf,%lf,%lf\n",axisx[i], axisy[i], forcez[i]);
		}
		fclose(fpw);
	}
};

//�ǂݍ��݃t�@�C���I�[�v���i�t�H�[�X�v���[�g�j
bool openFiles(FILE* &fp, char* readfileName, int subNumber, int taskNumber, int triNumber, int plate_id){
	sprintf(readfileName,"D:/User/kanai/Data/%s/sub%d/csv/fp/%s%04d%_f_%d.csv", datafile, subNumber + 1, taskName[taskNumber], triNumber + 1, plate_id + 1);	//�ǂݍ��݃t�@�C�����ݒ�
	if((fp = fopen(readfileName,"r")) == NULL){
		printf("%s open error\n",readfileName);
		return false;
	} else {
		printf("open %s\n", readfileName);
	}
	return true;
}
//���x���폜
void removeHeader(FILE* &fp){
	char buf[1024];
	fscanf(fp,"%s%s%s%s%s%s%s%s%s",&buf,&buf,&buf,&buf,&buf,&buf,&buf,&buf,&buf);
}

//�v���f�[�^���
void loadData(FILE* &fp, ForceplateData *dist){
	char buf[1024];
	fscanf(fp,"%s",buf);
	sscanf(buf,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &dist->forcetime,&dist->forcex,&dist->forcey,&dist->forcez,&dist->momentx,&dist->momenty,&dist->momentz,&dist->momentdashx,&dist->momentdashy,&dist->axisx,&dist->axisy,&dist->torque,&dist->cofx,&dist->cofy,&dist->cofz);	//�������琔�l�ɕϊ�
}

int main(void){

	CenterData centerData;//������COP
	ForceplateData frontData;//�O�ɏo���Ă鑫�̂ق��̃f�[�^
	ForceplateData rearData;//��둤�̑��̃f�[�^

	EvaluationIndex indexdata;		//�]���w�W�f�[�^

	char readfilename[MAXTEXT];				//�ǂݍ��݃t�@�C����
	char writefilename[MAXTEXT];			//�������݃t�@�C����(�e����)

	FILE *fpFront, *fpRear;					//�ǂݍ��݃t�@�C���i�t�H�[�X�v���[�g�j
	FILE *fpw;								//�������݃t�@�C��

	int subnumber;						//�팱�Ґ�
	int tasknumber;						//�^�X�N��
	int trinumber;						//���s��

	for(subnumber = 0; subnumber<SUBNUM; subnumber++){

		//�������݃t�@�C���I�[�v��(�ŏI����)
		sprintf(writefilename,"D:/User/kanai/Data/%s/result_COP/result/sub%dresult.csv", datafile, subnumber + 1);		//�������݃t�@�C�����ݒ�
		if((fpw = fopen(writefilename,"w")) == NULL){
			printf("endresult open error. cannot open%s\n", writefilename);
			continue;
		}

		for(tasknumber = 0; tasknumber<TASKNUM; tasknumber++){

			//���x����������
			fprintf(fpw,"�^�X�N�ԍ�,");
			fprintf(fpw,"���s��,");
			fprintf(fpw,"�]�|�L��,holdtime,������,");
			fprintf(fpw,"���O�Ւ�x,���O�Ւ�y,���O�Ւ�,");
			fprintf(fpw,"�P�ʋO�Ւ�x,�P�ʋO�Ւ�y,�P�ʋO�Ւ�,");
			fprintf(fpw,"SDx,SDy,���s�l,");
			fprintf(fpw,"�U����x,�U����y,");
			fprintf(fpw,"��`�ʐ�,���s�l�ʐ�,�W���΍��ʐ�,");
			fprintf(fpw,"\n");

			for(trinumber = 0; trinumber<TRIALNUM; trinumber++){

				if(!openFiles(fpRear, readfilename, subnumber, tasknumber, trinumber, 0)){
					return -1;
				}
				if(!openFiles(fpFront, readfilename, subnumber, tasknumber, trinumber, 1)){
					return -1;
				}

				//�t�@�C���̃w�b�_���폜
				removeHeader(fpRear);
				removeHeader(fpFront);

				centerData.clear();

				while(1){
					loadData(fpRear, &rearData);
					loadData(fpFront, &frontData);

					centerData.set(frontData, rearData);

					if(centerData.isFailure())	break;

					//�t�@�C���[������
					if(feof(fpFront) != 0)							break;
				}

				/////////////////////////////////////////////
				//�]���w�W�Z�o
				/////////////////////////////////////////////
				indexdata = centerData.getEvalationIndex();

				//check centerData
				centerData.dump(subnumber, tasknumber, trinumber);

				//���ʋL�q
				printf("Sub.%d �^�X�N%d ���s%d�̉�͏I���I\n",subnumber+1,tasknumber+1,trinumber+1);
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
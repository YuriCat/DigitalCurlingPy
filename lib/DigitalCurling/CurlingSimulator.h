
#include <deque>
#include <iostream>

#include <random>

// ゲーム情報
typedef struct _GAMESTATE{
	int		ShotNum;		// 現在のショット数
	// ShotNum が n の場合、次に行うショットが n+1 投目になる
	
	int		CurEnd;			// 現在のエンド数
	int		LastEnd;		// 最終エンド数
	int		Score[10];		// 第1エンドから第10エンドまでのスコア
	bool	WhiteToMove;	// 手番の情報
	// WhiteToMove が 0 の場合次のショットを行うのは先手、WhiteToMove が 1 の場合次のショットを行うのは後手となる
	
	float	body[16][2];	// body[n] は n 投目のストーンの位置座標を表す
	// body[n][0] は n 投目のストーンの x 座標、body[n][1] は n 投目のストーンの y 座標を表す
	
} GAMESTATE, *PGAMESTATE;

// ショット情報（座標）
typedef struct _ShotPos{
	float x;
	float y;
	bool angle;
	
} SHOTPOS, PSHOTPOS;

// ショット情報（強さベクトル）
typedef struct _ShotVec{
	float x;
	float y;
	bool angle;
	
} SHOTVEC, PSHOTVEC;

// ストーン情報
#define STONEINFO_SIZE 0.145f

// 不要な場合は以下コメントにしてください

/* シミュレーション関数 */
// GAMESTATE	*pGameState	- シミュレーション前の局面情報
// SHOTVEC		Shot		- シミュレーションを行うショットベクトル
// float		Rand		- 乱数の大きさ
// SHOTVEC		*lpResShot	- 実際にシミュレーションで使われたショットベクトル（乱数が加えられたショットベクトルの値）
// int			LoopCount	- 何フレームまでシミュレートを行うか（-1を指定すると最後までシミュレーションを行う）
// 戻り値	- Simulation関数が失敗すると0が返ります。成功するとそれ以外の値が返ります
int Simulation( GAMESTATE *pGameState, SHOTVEC Shot, float Rand = STONEINFO_SIZE, SHOTVEC *lpResShot = nullptr, int LoopCount = -1 );

/* シミュレーション関数（拡張版） */
// GAMESTATE	*pGameState	- シミュレーション前の局面情報
// SHOTVEC		Shot		- シミュレーションを行うショットベクトル
// float		RandX		- 横方向の乱数の大きさ
// float		RandY		- 縦方向の乱数の大きさ
// SHOTVEC		*lpResShot	- 実際にシミュレーションで使われたショットベクトル（乱数が加えられたショットベクトルの値）
// float		*pLoci		- シミュレーション結果（軌跡）を受け取る配列
//							  1フレーム毎のストーンの位置座標（X,Y合わせ32個セット）が一次元配列で返る
// int			ResLociSize	- シミュレーション結果（軌跡）を受け取る配列の最大サイズ（シミュレーション結果がこのサイズを超える場合には、このサイズまで結果が格納されます）
// 戻り値	- SimulationEx関数が失敗すると0が返ります。成功した場合にはシミュレーションに掛ったフレーム数が返ります
int SimulationEx( GAMESTATE *pGameState, SHOTVEC Shot, float RandX, float RandY, SHOTVEC *lpResShot, float *pLoci, int ResLociSize );

/* ショット生成関数（ドローショット） */
// SHOTPOS		ShotPos			- 座標を指定します。ここで指定した座標で止まるショットが生成されます
// SHOTVEC		*lpResShotVec	- 生成されたショットを受け取るアドレスを指定します
// 戻り値	- CreateShot関数が失敗すると0が返ります。成功するとそれ以外の値が返ります
int CreateShot( SHOTPOS ShotPos, SHOTVEC *lpResShotVec );

/* ショット生成関数（テイクショット） */
// SHOTPOS		ShotPos			- 座標を指定します。ここで指定した座標を通るショットが生成されます
// float		Power			- ショットの強さを指定します。ここで指定した強さのショットが生成されます
// SHOTVEC		*lpResShotVec	- 生成されたショットを受け取るアドレスを指定します
// 戻り値	- CreateShot関数が失敗すると0が返ります。成功するとそれ以外の値が返ります
int CreateHitShot( SHOTPOS Shot, float Power, SHOTVEC *lpResShot );

//typedef int (*SIMULATION_FUNC)( GAMESTATE *pGameState, SHOTVEC Shot, float Rand, SHOTVEC *lpResShot, int LoopCount );
//typedef int (*SIMULATIONEX_FUNC)( GAMESTATE *pGameState, SHOTVEC Shot, float RandX, float RandY, SHOTVEC *lpResShot, float *pLoci, size_t ResLociSize );
//typedef int (*CREATESHOT_FUNC)( SHOTPOS ShotPos, SHOTVEC *lpResShotVec );
//typedef int (*CREATEHITSHOT_FUNC)( SHOTPOS Shot, float Power, SHOTVEC *lpResShot );

extern std::mt19937 dice;

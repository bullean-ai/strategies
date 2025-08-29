package strategies

import (
	"fmt"
	"log"
	"reflect"
	"sync"

	binanceDomain "github.com/bullean-ai/bullean-go/binance/domain"
	"github.com/bullean-ai/bullean-go/data"
	"github.com/bullean-ai/bullean-go/data/domain"
	"github.com/bullean-ai/bullean-go/indicators"
	"github.com/bullean-ai/bullean-go/neural_nets"
	ffnnDomain "github.com/bullean-ai/bullean-go/neural_nets/domain"
	"github.com/bullean-ai/bullean-go/neural_nets/ffnn"
	"github.com/bullean-ai/bullean-go/neural_nets/ffnn/solver"
	"github.com/bullean-ai/bullean-go/strategies"
	buySellStrategy "github.com/bullean-ai/bullean-go/strategies/domain"
	domain2 "github.com/bullean-ai/strategies/strategies/domain"
)

type TrainingTask struct {
	examples ffnnDomain.Examples
}

type AIStrategyV1 struct {
	BaseAsset     string
	TradeAsset    string
	QuoteAsset    string
	candles       []domain.Candle
	NeuralNetConf *ffnnDomain.Config
	inputLen      int
	ranger        int
	iterations    int
	lr            float64

	trainingModel *ffnn.FFNN

	mu              sync.RWMutex
	activeModel     *ffnn.FFNN
	activeEvaluator *neural_nets.Evaluator

	strategy       *strategies.Strategy
	lastprediction int
	longPosExist   bool
	shortPosExist  bool
	isReady        bool

	trainingChan chan TrainingTask
}

func NewAIStrategyV1(input_len int, ranger int, iterations int, lr float64, config ffnnDomain.Config) domain2.IStrategyModel {
	neuralNetConf := config

	// Modelleri ve değerlendiricileri bir kere başlat
	trainingModel := ffnn.NewFFNN(neuralNetConf)
	activeModel := ffnn.NewFFNN(neuralNetConf)
	activeEvaluator := neural_nets.NewEvaluator([]ffnnDomain.Neural{
		{
			Model: activeModel,
		},
	})

	s := &AIStrategyV1{
		NeuralNetConf:   &neuralNetConf,
		inputLen:        input_len,
		ranger:          ranger,
		iterations:      iterations,
		lr:              lr,
		trainingModel:   trainingModel,
		activeModel:     activeModel,
		activeEvaluator: activeEvaluator,
		lastprediction:  0,
		longPosExist:    false,
		shortPosExist:   false,
		isReady:         false,

		trainingChan: make(chan TrainingTask, 1),
	}

	go s.runTrainer()

	return s
}

func (st *AIStrategyV1) runTrainer() {
	trainer := ffnn.NewBatchTrainer(solver.NewAdam(st.lr, 0, 0, 1e-12), 1, 100, 6)

	for task := range st.trainingChan {
		st.trainingModel = ffnn.NewFFNN(*st.NeuralNetConf)
		evaluator := neural_nets.NewEvaluator([]ffnnDomain.Neural{
			{
				Model:      st.trainingModel,
				Trainer:    trainer,
				Iterations: st.iterations,
			},
		})
		evaluator.Train(task.examples, task.examples)
		log.Println("Eğitim tamamlandı.")
		st.mu.Lock()
		st.activeModel = st.trainingModel
		st.activeEvaluator = neural_nets.NewEvaluator([]ffnnDomain.Neural{
			{
				Model: st.activeModel,
			},
		})
		st.mu.Unlock()
	}
}

func (st *AIStrategyV1) Init(base_asset, trade_asset, quote_asset string, binanceClients []binanceDomain.IBinanceClient, candles []domain.Candle, is_ready func(mapName string, is_ready bool)) {
	strategy := strategies.NewStrategy(base_asset, trade_asset, []string{quote_asset}, 40, binanceClients)
	st.strategy = strategy
	st.BaseAsset = base_asset
	st.TradeAsset = trade_asset
	st.QuoteAsset = quote_asset
	st.candles = candles
	strategy.BinanceClients = binanceClients

	pair := fmt.Sprintf("%s%s", st.QuoteAsset, st.BaseAsset)
	structName := reflect.TypeOf(st).Name()
	mapName := fmt.Sprintf("%s_%s", structName, pair)

	examples := st.createExamples(candles)

	st.trainingChan <- TrainingTask{
		examples: examples,
	}

	st.isReady = true
	is_ready(mapName, true)
	return
}

// OnCandle, yeni mum verisi geldiğinde çağrılır
func (st *AIStrategyV1) OnCandle(candle domain.Candle) {
	pair := fmt.Sprintf("%s%s", st.QuoteAsset, st.BaseAsset)

	if len(st.candles) > 0 {
		st.candles = st.candles[1:]
		st.candles = append(st.candles, candle)
	}

	if st.isReady == false {
		return
	}

	examples := st.createExamples(st.candles)

	select {
	case st.trainingChan <- TrainingTask{examples: examples}:
	default:
	}

	// Tahmin işlemi için okuma kilidi kullandık
	st.mu.RLock()
	if st.activeEvaluator == nil || len(examples) == 0 {
		st.mu.RUnlock()
		return
	}
	pred := st.activeEvaluator.Predict(examples[len(examples)-1].Input)
	st.mu.RUnlock()

	prediction := 0
	buy := pred[0]
	sell := pred[1]
	hold := pred[2]

	if buy >= .6 {
		prediction = 1
	} else if sell >= .6 {
		prediction = -1
	} else if hold >= .6 {
		prediction = 0
	}

	st.strategy.Next(map[string]domain.Candle{
		pair: candle,
	})

	st.strategy.Evaluate(func(lastLongEnterPrice, lastLongClosePrice float64) buySellStrategy.PositionType {
		if prediction == 1 && st.lastprediction == 1 && !st.longPosExist {
			st.longPosExist = true
			return buySellStrategy.POS_BUY
		} else if prediction == -1 && st.lastprediction == -1 {
			st.longPosExist = false
			return buySellStrategy.POS_SELL
		} else {
			return buySellStrategy.POS_HOLD
		}
	}, func(lastShortEnterPrice, lastShortClosePrice float64) buySellStrategy.PositionType {
		if lastShortEnterPrice == 0 {
			lastShortEnterPrice = candle.Close
		}
		if prediction == -1 && st.lastprediction == -1 && !st.shortPosExist {
			st.shortPosExist = true
			return buySellStrategy.POS_BUY
		} else if prediction == 1 && st.lastprediction == 1 {
			st.shortPosExist = false
			return buySellStrategy.POS_SELL
		} else {
			return buySellStrategy.POS_HOLD
		}
	})

	st.lastprediction = prediction
}

func (st *AIStrategyV1) UpdateBinanceClients(binance_clients []binanceDomain.IBinanceClient) {
	if st.strategy == nil {
		fmt.Println("Strategy is not initialized, cannot update Binance clients")
		return
	}
	st.strategy.BinanceClients = binance_clients
}

func (st *AIStrategyV1) createExamples(candles []domain.Candle) ffnnDomain.Examples {
	var examples ffnnDomain.Examples
	dataset := data.NewDataSet(candles, st.inputLen)

	dataset.CreatePolicy(domain.PolicyConfig{
		FeatName:    "feature_per_change",
		FeatType:    domain.FEAT_CLOSE_PERCENTAGE_TRADE_TYPE,
		PolicyRange: st.ranger,
	}, func(candles []domain.Candle) int {
		ema := indicators.MA(candles, 50)
		if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= 0.3 {
			return 1
		} else if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) < 0.3 && buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= 0 {
			return 0
		} else if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) < 0 && buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= -0.3 {
			return 0
		} else {
			return 2
		}
	})
	dataset.SerializeLabels()
	dataFrame := dataset.GetDataSet()

	for i := 0; i < len(dataFrame); i++ {
		label := []float64{}
		if dataFrame[i].Label == 1 {
			label = []float64{1, 0, 0}
		} else if dataFrame[i].Label == 2 {
			label = []float64{0, 1, 0}
		} else {
			label = []float64{0, 0, 1}
		}
		examples = append(examples, ffnnDomain.Example{
			Input:    dataFrame[i].Features,
			Response: label,
		})
	}
	return examples
}

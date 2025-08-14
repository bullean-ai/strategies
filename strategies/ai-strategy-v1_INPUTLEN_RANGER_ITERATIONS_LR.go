package strategies

import (
	"fmt"
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
	"reflect"
)

type AIStrategyV1 struct {
	BaseAsset         string
	TradeAsset        string
	QuoteAsset        string
	candles           []domain.Candle
	NeuralNetConf     *ffnnDomain.Config
	inputLen          int
	ranger            int
	iterations        int
	lr                float64
	trainingModel     *ffnn.FFNN
	activeModel       *ffnn.FFNN
	activeEvaluator   *neural_nets.Evaluator
	trainingEvaluator *neural_nets.Evaluator
	strategy          *strategies.Strategy

	lastprediction int
	isTrainingEnd  bool
	longPosExist   bool
	shortPosExist  bool
	isReady        bool
}

func NewAIStrategyV1(input_len int, ranger int, iterations int, lr float64, config *ffnnDomain.Config) domain2.IStrategyModel {
	neuralNetConf := config
	trainingModel := ffnn.NewFFNN(neuralNetConf)
	activeModel := ffnn.NewFFNN(neuralNetConf)
	trainingEvaluator := neural_nets.NewEvaluator([]ffnnDomain.Neural{
		{
			Model:      trainingModel,
			Trainer:    ffnn.NewBatchTrainer(solver.NewAdam(lr, 0, 0, 1e-12), 1, 100, 12),
			Iterations: iterations,
		},
	})
	activeEvaluator := neural_nets.NewEvaluator([]ffnnDomain.Neural{
		{
			Model:      trainingModel,
			Trainer:    ffnn.NewBatchTrainer(solver.NewAdam(lr, 0, 0, 1e-12), 1, 100, 12),
			Iterations: iterations,
		},
	})

	return &AIStrategyV1{
		NeuralNetConf:     neuralNetConf,
		inputLen:          input_len,
		ranger:            ranger,
		trainingModel:     trainingModel,
		activeModel:       activeModel,
		trainingEvaluator: trainingEvaluator,
		activeEvaluator:   activeEvaluator,
		iterations:        iterations,
		lr:                lr,
		lastprediction:    0,
		isTrainingEnd:     true,
		longPosExist:      false,
		shortPosExist:     false,
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
	var examples ffnnDomain.Examples

	dataset := data.NewDataSet(candles, st.inputLen)

	dataset.CreatePolicy(domain.PolicyConfig{
		FeatName:    "feature_per_change",
		FeatType:    domain.FEAT_CLOSE_PERCENTAGE_TRADE_TYPE,
		PolicyRange: st.ranger,
	}, func(candles []domain.Candle) int {
		ema := indicators.MA(candles, 50)
		if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= 0.5 {
			return 1
		} else if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) < 0.5 && buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= 0 {
			return 0
		} else if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) < 0 && buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= -0.5 {
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
	st.trainingEvaluator.Train(examples, examples)
	is_ready(mapName, true)
	st.isReady = true
	return
}

func (st *AIStrategyV1) OnCandle(candle domain.Candle) {
	pair := fmt.Sprintf("%s%s", st.QuoteAsset, st.BaseAsset)
	var examples ffnnDomain.Examples
	var prediction int
	if len(st.candles) > 0 {
		st.candles = st.candles[1:]
		st.candles = append(st.candles, candle)
	}

	if st.isReady == false {
		return
	}
	dataset := data.NewDataSet(st.candles, st.inputLen)

	dataset.CreatePolicy(domain.PolicyConfig{
		FeatName:    "feature_per_change",
		FeatType:    domain.FEAT_CLOSE_PERCENTAGE_TRADE_TYPE,
		PolicyRange: st.ranger,
	}, func(candles []domain.Candle) int {
		ema := indicators.MA(candles, 50)
		if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= 0.5 {
			return 1
		} else if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) < 0.5 && buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= 0 {
			return 0
		} else if buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) < 0 && buySellStrategy.PercentageChange(ema[0], ema[len(ema)-1]) >= -0.5 {
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
	go func() {
		if st.isTrainingEnd {
			st.isTrainingEnd = false
			model2 := ffnn.NewFFNN(st.NeuralNetConf /*ffnnDomain.DefaultFFNNConfig(ranger)*/)
			newEvaluator := neural_nets.NewEvaluator([]ffnnDomain.Neural{
				{
					Model:      model2,
					Trainer:    ffnn.NewBatchTrainer(solver.NewAdam(st.lr, 0, 0, 1e-12), 1, 100, 12),
					Iterations: st.iterations,
				},
			})
			newEvaluator.Train(examples, examples)
			st.activeModel = model2
			newEvaluator = neural_nets.NewEvaluator([]ffnnDomain.Neural{
				{
					Model:      st.activeModel,
					Trainer:    ffnn.NewBatchTrainer(solver.NewAdam(st.lr, 0, 0, 1e-12), 1, 100, 12),
					Iterations: st.iterations,
				},
			})
			st.activeEvaluator = newEvaluator
			st.isTrainingEnd = true
		}
	}()
	pred := st.activeEvaluator.Predict(examples[len(examples)-1].Input)
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
	st.strategy.Evaluate(func(lastLongEnterPrice, lastLongClosePrice float64) buySellStrategy.PositionType { // Long Enter
		if prediction == 1 && st.lastprediction == 1 && !st.longPosExist {
			st.longPosExist = true
			return buySellStrategy.POS_BUY
		} else if prediction == -1 && st.lastprediction == -1 {
			st.longPosExist = false
			return buySellStrategy.POS_SELL
		} else {
			return buySellStrategy.POS_HOLD
		}

	}, func(lastShortEnterPrice, lastShortClosePrice float64) buySellStrategy.PositionType { // Short Enter
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

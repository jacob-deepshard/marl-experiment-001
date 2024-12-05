import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import _ from "lodash";

// Create a window moving average function for smoothing
const movingAverage = (data, windowSize) => {
  return data.map((val, idx, arr) => {
    const start = Math.max(0, idx - windowSize + 1);
    const window = arr.slice(start, idx + 1);
    return window.reduce((a, b) => a + b, 0) / window.length;
  });
};

const SwarmRLComparison = () => {
  const [data, setData] = useState([]);
  const [isTraining, setIsTraining] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let mounted = true;

    const runTraining = async () => {
      const env = new Environment();
      const trainer = new Trainer(env, {
        batchSize: 64,
        numEpisodes: 100, // Reduced for visualization
        maxSteps: 200,
        policyLR: 0.0003,
        valueLR: 0.001,
        gamma: 0.99,
        lambda: 0.95,
        entropyCoef: 0.01,
        ppoEpsilon: 0.2,
      });

      const trainingLoop = async () => {
        let episode = 0;
        const results = [];

        while (episode < trainer.config.numEpisodes && mounted && isTraining) {
          // Run one episode for each algorithm
          const clResult = await trainer.runEpisode(
            trainer.clPolicy,
            trainer.clValue,
            false
          );
          const mclResult = await trainer.runEpisode(
            trainer.mclPolicy,
            trainer.mclValue,
            true
          );

          // Store results
          results.push({
            episode,
            clReward: clResult.totalReward,
            mclReward: mclResult.totalReward,
            clLoss: clResult.loss,
            mclLoss: mclResult.loss,
            clValue: clResult.averageValue,
            mclValue: mclResult.averageValue,
          });

          // Update state with smoothed data
          if (mounted) {
            const smoothedData = results.map((d, i) => ({
              ...d,
              clRewardSmooth: movingAverage(
                results.slice(0, i + 1).map((r) => r.clReward),
                10
              )[i],
              mclRewardSmooth: movingAverage(
                results.slice(0, i + 1).map((r) => r.mclReward),
                10
              )[i],
            }));
            setData(smoothedData);
            setProgress(((episode + 1) / trainer.config.numEpisodes) * 100);
          }

          episode++;
          // Small delay to allow React to update
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      };

      try {
        await trainingLoop();
      } finally {
        if (mounted) {
          setIsTraining(false);
        }
      }
    };

    runTraining();

    return () => {
      mounted = false;
      setIsTraining(false);
    };
  }, []);

  return (
    <div className="space-y-4">
      <Card className="w-full">
        <CardHeader>
          <CardTitle>
            Swarm RL Training Progress: {progress.toFixed(1)}%
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-8">
            {/* Rewards Chart */}
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="clRewardSmooth"
                    name="Cross Learning Reward"
                    stroke="#8884d8"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="mclRewardSmooth"
                    name="Maynard-Cross Learning Reward"
                    stroke="#82ca9d"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Loss Chart */}
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="clLoss"
                    name="Cross Learning Loss"
                    stroke="#ff7c43"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="mclLoss"
                    name="Maynard-Cross Learning Loss"
                    stroke="#f95d6a"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Value Estimates Chart */}
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="episode" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="clValue"
                    name="Cross Learning Value"
                    stroke="#003f5c"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="mclValue"
                    name="Maynard-Cross Learning Value"
                    stroke="#665191"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SwarmRLComparison;

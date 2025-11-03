package org.example;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

/**
 * Java + DJL inference example for TorchScript EdgeClassifierWrapper.
 * Generates node features and edge features only.
 */
public class Main {

    /** Graph input data structure. */
    static class GraphInput {
        float[][] x;
        long[][] edgeIndex;
        float[][] edgeAttr;

        public GraphInput(float[][] x, long[][] edgeIndex, float[][] edgeAttr) {
            this.x = x;
            this.edgeIndex = edgeIndex;
            this.edgeAttr = edgeAttr;
        }
    }

    public static void main(String[] args) {

        String csvFile = "clusters_sector1_small.csv";
        int maxEvents = 20;

        // --- 改动 1：扩展 avgWire_diff_max ---
        float[][] avgWireDiffMax = {
            {12f, 20f, 12f, 38f, 14f},   // |ΔSL| = 1
            {14f, 18f, 40f, 40f},        // |ΔSL| = 2
            {24f, 48f, 48f},             // |ΔSL| = 3
            {55f, 55f},                  // |ΔSL| = 4
            {56f}                        // |ΔSL| = 5
        };

        boolean bidirectional = true;
        String modelPath = "nets/gnn_default.pt"; // TorchScript model path

        try {
            // -----------------------------
            // Load graphs from CSV
            // -----------------------------
            List<GraphInput> graphs = loadGraphs(csvFile, maxEvents, avgWireDiffMax, bidirectional);
            System.out.println("Loaded " + graphs.size() + " graph(s)");

            // -----------------------------
            // Translator for TorchScript model
            // -----------------------------
            Translator<GraphInput, float[]> translator = new Translator<>() {
                @Override
                public NDList processInput(TranslatorContext ctx, GraphInput input) {
                    NDManager manager = ctx.getNDManager();
                    NDArray xNd = manager.create(input.x);
                    NDArray edgeNd = manager.create(input.edgeIndex);
                    NDArray edgeAttrNd = manager.create(input.edgeAttr);
                    return new NDList(xNd, edgeNd, edgeAttrNd);
                }

                @Override
                public float[] processOutput(TranslatorContext ctx, NDList list) {
                    return list.get(0).toFloatArray();
                }

                @Override
                public Batchifier getBatchifier() {
                    return null; // no batching
                }
            };

            // -----------------------------
            // Load TorchScript model
            // -----------------------------
            Criteria<GraphInput, float[]> criteria = Criteria.builder()
                    .setTypes(GraphInput.class, float[].class)
                    .optModelPath(Paths.get(modelPath))
                    .optEngine("PyTorch")
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .build();

            try (ZooModel<GraphInput, float[]> model = criteria.loadModel();
                 Predictor<GraphInput, float[]> predictor = model.newPredictor()) {

                // -----------------------------
                // Inference
                // -----------------------------
                for (int i = 0; i < graphs.size(); i++) {
                    GraphInput g = graphs.get(i);
                    float[] preds = predictor.predict(g);
                    System.out.printf("Event %d: Nodes=%d, Edges=%d, Output=%d%n",
                            i + 1, g.x.length, g.edgeIndex[0].length, preds.length);
                }
            }

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException("Inference failed: " + e.getMessage(), e);
        }
    }

    // -----------------------------
    // CSV parsing and candidate edge generation
    // -----------------------------
    private static List<GraphInput> loadGraphs(String csvFile, int maxEvents, float[][] avgWireDiffMax, boolean bidirectional) throws IOException {
        List<GraphInput> graphs = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            int currentEvent = -1;
            List<Float> avgWireList = new ArrayList<>();
            List<Float> slopeList = new ArrayList<>();
            List<Integer> superlayers = new ArrayList<>();

            while ((line = br.readLine()) != null && graphs.size() < maxEvents) {
                if (line.startsWith("eventIdx")) continue;
                String[] tokens = line.split(",");
                int eventIdx = Integer.parseInt(tokens[0]);
                float avgWire = Float.parseFloat(tokens[2]);
                float slope = Float.parseFloat(tokens[3]);
                int superlayer = Integer.parseInt(tokens[4]);

                if (eventIdx != currentEvent) {
                    if (currentEvent != -1) {
                        graphs.add(generateCandidateEdges(avgWireList, slopeList, superlayers, avgWireDiffMax, bidirectional));
                    }
                    currentEvent = eventIdx;
                    avgWireList.clear();
                    slopeList.clear();
                    superlayers.clear();
                }

                // --- 改动 2：节点特征包括 slope ---
                // Node feature: [avgWire_norm, superlayer_norm, slope]
                float[] feat = new float[3];
                feat[0] = avgWire;
                feat[1] = slope; // 不归一化
                feat[2] = superlayer;

                avgWireList.add(avgWire);
                slopeList.add(slope);
                superlayers.add(superlayer);
            }

            if (!avgWireList.isEmpty() && graphs.size() < maxEvents) {
                graphs.add(generateCandidateEdges(avgWireList, slopeList, superlayers, avgWireDiffMax, bidirectional));
            }
        }

        return graphs;
    }

    private static GraphInput generateCandidateEdges(
            List<Float> avgWireList,
            List<Float> slopeList,
            List<Integer> superlayers,
            float[][] avgWireDiffMax,
            boolean bidirectional
    ) {
        int n = avgWireList.size();
        float avgWireRange = 112.0f;
        float superlayerRange = 6.0f;

        // 构建节点特征
        float[][] x = new float[n][3];
        for (int i = 0; i < n; i++) {
            x[i][0] = avgWireList.get(i) / avgWireRange;   // avgWire_norm
            x[i][1] = slopeList.get(i);                    // slope
            x[i][2] = superlayers.get(i) / superlayerRange; // superlayer_norm
        }

        // --- 改动 3：扩展 ΔSL 判断逻辑 ---
        List<long[]> edges = new ArrayList<>();
        List<float[]> edgeAttrList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int sl_i = superlayers.get(i);
                int sl_j = superlayers.get(j);
                int slDiff = sl_i - sl_j;
                int absSlDiff = Math.abs(slDiff);
                float avgWireDiff = avgWireList.get(i) - avgWireList.get(j);

                boolean validEdge = false;
                if (absSlDiff >= 1 && absSlDiff <= avgWireDiffMax.length) {
                    int row = absSlDiff - 1;
                    int col = Math.min(sl_i, sl_j) - 1;
                    if (col < avgWireDiffMax[row].length &&
                        Math.abs(avgWireDiff) < avgWireDiffMax[row][col]) {
                        validEdge = true;
                    }
                }

                if (validEdge) {
                    float slopeDiff = slopeList.get(i) - slopeList.get(j);
                    edges.add(new long[]{i, j});
                    edgeAttrList.add(new float[]{avgWireDiff / avgWireRange, slopeDiff, slDiff / superlayerRange});

                    if (bidirectional) {
                        edges.add(new long[]{j, i});
                        edgeAttrList.add(new float[]{-avgWireDiff / avgWireRange, -slopeDiff, -slDiff / superlayerRange});
                    }
                }
            }
        }

        float[][] edgeAttr = edgeAttrList.toArray(new float[0][]);
        long[][] edgeIndex = new long[2][edges.size()];
        for (int k = 0; k < edges.size(); k++) {
            edgeIndex[0][k] = edges.get(k)[0];
            edgeIndex[1][k] = edges.get(k)[1];
        }

        return new GraphInput(x, edgeIndex, edgeAttr);
    }
}

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
public class CandidateEdgeGenerator {

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

        String csvFile = "clusters.csv";
        int maxEvents = 20;
        float[][] maxWireDiff = {
            {12f, 20f, 12f, 38f, 14f, 0f},   // superlayer difference = 1
            {14f, 18f, 40f, 40f, 0f, 0f}     // superlayer difference = 2
        };
        boolean bidirectional = true;
        String modelPath = "model/edge_classifier.pt"; // TorchScript model path

        try {
            // -----------------------------
            // Load graphs from CSV
            // -----------------------------
            List<GraphInput> graphs = loadGraphs(csvFile, maxEvents, maxWireDiff, bidirectional);
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
                            i, g.x.length, g.edgeIndex[0].length, preds.length);
                }
            }

        } catch (IOException | ModelNotFoundException | MalformedModelException | TranslateException e) {
            throw new RuntimeException("Inference failed: " + e.getMessage(), e);
        }
    }

    // -----------------------------
    // CSV parsing and candidate edge generation (unchanged)
    // -----------------------------
    private static List<GraphInput> loadGraphs(String csvFile, int maxEvents, float[][] avgWireDiffMax, boolean bidirectional) throws IOException {
        List<GraphInput> graphs = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            int currentEvent = -1;
            List<float[]> xList = new ArrayList<>();
            List<Float> avgWireList = new ArrayList<>();
            List<Integer> superlayers = new ArrayList<>();

            while ((line = br.readLine()) != null && graphs.size() < maxEvents) {
                if (line.startsWith("eventIdx")) continue;
                String[] tokens = line.split(",");
                int eventIdx = Integer.parseInt(tokens[0]);
                float avgWire = Float.parseFloat(tokens[2]);
                int superlayer = Integer.parseInt(tokens[3]);

                if (eventIdx != currentEvent) {
                    if (currentEvent != -1 && !xList.isEmpty()) {
                        graphs.add(generateCandidateEdges(xList, avgWireList, superlayers, avgWireDiffMax, bidirectional));
                    }
                    currentEvent = eventIdx;
                    xList.clear();
                    avgWireList.clear();
                    superlayers.clear();
                }

                // Node feature: [avgWire normalized, superlayer one-hot]
                float[] feat = new float[7];
                feat[0] = avgWire;
                if (superlayer >= 1 && superlayer <= 6) feat[superlayer] = 1.0f;

                xList.add(feat);
                avgWireList.add(avgWire);
                superlayers.add(superlayer);
            }

            if (!xList.isEmpty() && graphs.size() < maxEvents) {
                graphs.add(generateCandidateEdges(xList, avgWireList, superlayers, avgWireDiffMax, bidirectional));
            }
        }

        return graphs;
    }

    private static GraphInput generateCandidateEdges(
            List<float[]> xList,
            List<Float> avgWireList,
            List<Integer> superlayers,
            float[][] avgWireDiffMax,
            boolean bidirectional
    ) {
        List<long[]> edges = new ArrayList<>();
        List<float[]> edgeAttrList = new ArrayList<>();
        int n = xList.size();

        float avgWireMin = Collections.min(avgWireList);
        float avgWireMax = Collections.max(avgWireList);
        float avgWireRange = avgWireMax - avgWireMin;

        if (avgWireRange == 0) {
            System.out.println("Skipping event: avgWireRange == 0");
            return new GraphInput(xList.toArray(new float[0][]), new long[2][0], new float[0][]);
        }

        for (int i = 0; i < n; i++) {
            xList.get(i)[0] = (avgWireList.get(i) - avgWireMin) / avgWireRange;
        }

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int slDiff = superlayers.get(i) - superlayers.get(j);
                float avgWireDiff = avgWireList.get(i) - avgWireList.get(j);
                int absSlDiff = Math.abs(slDiff);

                boolean validEdge = false;
                if (absSlDiff == 1 && Math.abs(avgWireDiff) < avgWireDiffMax[0][superlayers.get(i)-1]) validEdge = true;
                else if (absSlDiff == 2 && Math.abs(avgWireDiff) < avgWireDiffMax[1][superlayers.get(i)-1]) validEdge = true;

                if (validEdge) {
                    edges.add(new long[]{i, j});
                    edgeAttrList.add(new float[]{slDiff, avgWireDiff / avgWireRange});
                    if (bidirectional) {
                        edges.add(new long[]{j, i});
                        edgeAttrList.add(new float[]{-slDiff, -avgWireDiff / avgWireRange});
                    }
                }
            }
        }

        float[][] x = xList.toArray(new float[0][]);
        float[][] edgeAttr = edgeAttrList.toArray(new float[0][]);
        long[][] edgeIndex = new long[2][edges.size()];
        for (int k = 0; k < edges.size(); k++) {
            edgeIndex[0][k] = edges.get(k)[0];
            edgeIndex[1][k] = edges.get(k)[1];
        }

        return new GraphInput(x, edgeIndex, edgeAttr);
    }
}

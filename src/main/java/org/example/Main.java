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
 * Java + DJL inference example for a TorchScript EdgeClassifierWrapper.
 * Generates node features and edge features for particle tracking.
 */
public class Main {

    /** Graph input data structure */
    static class GraphInput {
        float[][] x;
        long[][] edgeIndex;
        float[][] edgeAttr;

        /**
         * Constructor
         * @param x Node features: [num_nodes][num_features]
         * @param edgeIndex Edge indices: [2][num_edges]
         * @param edgeAttr Edge features: [num_edges][num_edge_features]
         */
        public GraphInput(float[][] x, long[][] edgeIndex, float[][] edgeAttr) {
            this.x = x;
            this.edgeIndex = edgeIndex;
            this.edgeAttr = edgeAttr;
        }
    }

    public static void main(String[] args) {

        String csvFile = "clusters_sector1_small.csv";
        int maxEvents = 20;

        // ΔSL maximum difference table
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
            // Load graphs from CSV
            List<GraphInput> graphs = loadGraphs(csvFile, maxEvents, avgWireDiffMax, bidirectional);
            System.out.println("Loaded " + graphs.size() + " graph(s)");

            // Translator for TorchScript model
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

            // Load TorchScript model
            Criteria<GraphInput, float[]> criteria = Criteria.builder()
                    .setTypes(GraphInput.class, float[].class)
                    .optModelPath(Paths.get(modelPath))
                    .optEngine("PyTorch")
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .build();

            try (ZooModel<GraphInput, float[]> model = criteria.loadModel();
                 Predictor<GraphInput, float[]> predictor = model.newPredictor()) {

                // Perform inference
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

    /**
     * Load graphs from CSV file and generate candidate edges.
     *
     * @param csvFile CSV file path containing cluster data
     * @param maxEvents Maximum number of events to load
     * @param avgWireDiffMax ΔSL maximum difference table
     * @param bidirectional Whether to generate bidirectional edges
     * @return List of GraphInput objects representing events
     * @throws IOException if CSV reading fails
     */
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

    /**
     * Generate candidate edges and edge attributes for a single event.
     *
     * @param avgWireList List of avgWire values for nodes
     * @param slopeList List of slope values for nodes
     * @param superlayers List of superlayer indices for nodes
     * @param avgWireDiffMax ΔSL maximum difference table
     * @param bidirectional Whether to generate bidirectional edges
     * @return GraphInput object containing node features, edge indices, and edge attributes
     */
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

        // Build node features
        float[][] x = new float[n][3];
        for (int i = 0; i < n; i++) {
            x[i][0] = avgWireList.get(i) / avgWireRange;   // avgWire normalized
            x[i][1] = slopeList.get(i);                    // slope
            x[i][2] = superlayers.get(i) / superlayerRange; // superlayer normalized
        }

        // Generate candidate edges
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

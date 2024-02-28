package com.example.facialemotionrecognition;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static final String TAG = "myLogs";
    private static final int REQUEST_CODE_SELECT_PICTURE = 1;
    private static final int REQUEST_CODE_PHOTO = 2;
    private static final String MODEL_FILE = "enet_b2_8_best.ptl";
    private static final String LABELS_FILE = "affectnet_labels.txt";

    List<String> labels;

    Module module;
    Bitmap selectedImage = null;
    ImageView ivPhoto;
    TextView tvResult;
    Button btnCamera;
    Button btnGallery;
    Button btnPredict;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ivPhoto = findViewById(R.id.ivPhoto);
        tvResult = findViewById(R.id.tvResult);
        btnCamera = findViewById(R.id.btnCamera);
        btnGallery = findViewById(R.id.btnGallery);
        btnPredict = findViewById(R.id.btnPredict);

        btnCamera.setOnClickListener(this);
        btnGallery.setOnClickListener(this);
        btnPredict.setOnClickListener(this);

        try {
            module = LiteModuleLoader.load(assetFilePath(this, MODEL_FILE));
        } catch (IOException e) {
            Log.e(TAG, "Unable to load model", e);
            throw new RuntimeException(e);
        }

        try {
            loadLabels(assetFilePath(this, LABELS_FILE));
        } catch (IOException e) {
            Log.e(TAG, "Unable to load labels", e);
            throw new RuntimeException(e);
        }
    }

    @Override
    public void onClick(View v) {
        int id = v.getId();
        if (id == R.id.btnGallery) {
            openImageFile(REQUEST_CODE_SELECT_PICTURE);
        } else if (id == R.id.btnPredict) {
            if (isImageLoaded()) {
                int predictionIdx = getPrediction(selectedImage);
                tvResult.setText("Predicted emotion: " + labels.get(predictionIdx));
                //Toast.makeText(this, labels.get(maxScoreIdx), Toast.LENGTH_SHORT).show();
            }
        } else if (id == R.id.btnCamera) {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(intent, REQUEST_CODE_PHOTO);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_CODE_SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                Log.d(TAG, "uri" + selectedImageUri);
                selectedImage = getImage(selectedImageUri);
            } else if (requestCode == REQUEST_CODE_PHOTO) {
                Log.d(TAG, "onActivityResult" + REQUEST_CODE_PHOTO);
                Bundle bndl = data.getExtras();
                if (bndl != null) {
                    Object obj = data.getExtras().get("data");
                    if (obj instanceof Bitmap) {
                        selectedImage = (Bitmap) obj;
                        Log.d(TAG, "bitmap " + selectedImage.getWidth() + " x "
                                + selectedImage.getHeight());
                    }
                }
            }
            if (selectedImage != null) {
                ivPhoto.setImageBitmap(selectedImage);
            }
        }
    }

    private int getPrediction(Bitmap image) {
        Bitmap scaledImage = Bitmap.createScaledBitmap(image, 260, 260, true);
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(scaledImage,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();

        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }
        return maxScoreIdx;
    }

    private void openImageFile(int requestCode) {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), requestCode);
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = Files.newOutputStream(file.toPath())) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    private void loadLabels(String filePath) throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        labels = new ArrayList<>();

        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(":", 2);
            String value = parts[1];
            labels.add(value);
        }

        reader.close();
    }

    private boolean isImageLoaded() {
        if (selectedImage == null)
            Toast.makeText(this, "It is necessary to open image firstly", Toast.LENGTH_SHORT).show();
        return selectedImage != null;
    }

    private Bitmap getImage(Uri selectedImageUri) {
        Bitmap bmp = null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            bmp = BitmapFactory.decodeStream(ims);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1);
            int degreesForRotation = 0;
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degreesForRotation = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degreesForRotation = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degreesForRotation = 180;
                    break;
            }
            if (degreesForRotation != 0) {
                Matrix matrix = new Matrix();
                matrix.setRotate(degreesForRotation);
                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
                        bmp.getHeight(), matrix, true);
            }

        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e + " " + Log.getStackTraceString(e));
        }
        return bmp;
    }
}
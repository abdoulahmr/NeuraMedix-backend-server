import json
import os
import base64 
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse, HttpResponseBadRequest, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
from scipy import stats
import random
from .models import UserNote

# Import your custom functions
from .molicule_binding import (
    calculate_similarity,
    calculate_molecular_descriptors,
    compare_functional_groups,
    generate_mol_image 
)
from .ihc_insight import (
    count_brown_pixels,
    brown_shade_distribution,
    generate_brown_mask_image_base64,
)
from .cell_detection import (
    detect_cells
)
from .research_analyzer import (
    analyze_research_paper
)
from .heart_disease_prediction import (
    predict_heart_disease
)
from .lung_cancer_prediction import (
    predict_pulmonary_disease
)
from .lung_cancer_detection import (
    predict_lung_cancer_from_image
)
from .auto_publish import generate_report_from_form

########################################################################################
#  Authentication and User Registration
########################################################################################

# User registration view
# This view allows users to register by providing a username, password, and email.
@csrf_exempt 
def register_user(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')

        if not username or not password:
            return JsonResponse({'error': 'Username and password are required'}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already exists'}, status=409)

        user = User.objects.create_user(username=username, password=password, email=email)
        return JsonResponse({'message': 'User registered successfully'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# JWT login view
# This view allows users to log in using their email and password, returning a JWT token upon
@csrf_exempt  
def jwt_login_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    data = json.loads(request.body.decode('utf-8'))
    email = data.get('email')
    password = data.get('password')
    print("Trying to log in with:", email, password)

    from django.contrib.auth import get_user_model
    User = get_user_model()

    try:
        user_obj = User.objects.get(email=email)
    except User.DoesNotExist:
        return JsonResponse({'error': 'User with this email does not exist'}, status=401)

    user = authenticate(username=user_obj.username, password=password)

    if user is not None:
        refresh = RefreshToken.for_user(user)
        return JsonResponse({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'username': user.username,
            'id': user.id
        })
    else:
        return JsonResponse({'error': 'Invalid credentials'}, status=401)

# Logout view
# This view allows users to log out by blacklisting their refresh token.
@api_view(['POST'])
def logout_view(request):
    try:
        data = json.loads(request.body)
        refresh_token = data.get("refresh")
        token = RefreshToken(refresh_token)
        token.blacklist()
        return Response({"message": "Logout successful"}, status=stats.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=stats.HTTP_400_BAD_REQUEST)

#########################################################################################
# User functions
#########################################################################################

# User notes view
# This view allows users to create, retrieve, update, and delete notes.
@api_view(['GET', 'POST', 'PUT', 'DELETE'])
@permission_classes([IsAuthenticated])
def user_notes_view(request):
    user = request.user
    if request.method == 'GET':
        notes = UserNote.objects.filter(user=user)
        notes_data = [{'id': note.id, 'content': note.content, 'created_at': note.created_at} for note in notes]
        return Response({'notes': notes_data})

    elif request.method == 'POST':
        data = request.data
        content = data.get('content')
        if not content:
            return Response({'error': 'Content is required.'}, status=400)
        note = UserNote.objects.create(user=user, content=content)
        return Response({'message': 'Note created', 'note': {'id': note.id, 'content': note.content, 'created_at': note.created_at}}, status=201)

    elif request.method == 'PUT':
        data = request.data
        note_id = data.get('id')
        content = data.get('content')
        if not note_id or not content:
            return Response({'error': 'ID and content are required.'}, status=400)
        try:
            note = UserNote.objects.get(id=note_id, user=user)
            note.content = content
            note.save()
            return Response({'message': 'Note updated', 'note': {'id': note.id, 'content': note.content, 'created_at': note.created_at}})
        except UserNote.DoesNotExist:
            return Response({'error': 'Note not found.'}, status=404)

    elif request.method == 'DELETE':
        data = request.data
        note_id = data.get('id')
        if not note_id:
            return Response({'error': 'ID is required.'}, status=400)
        try:
            note = UserNote.objects.get(id=note_id, user=user)
            note.delete()
            return Response({'message': 'Note deleted.'})
        except UserNote.DoesNotExist:
            return Response({'error': 'Note not found.'}, status=404)

    return Response({'error': 'Method not allowed.'}, status=405)
    
#########################################################################################
# Service Views
#########################################################################################

# Molecule binding prediction view
# This view accepts two SMILES strings, calculates their similarity, molecular descriptors,
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mol_binding_view(request):
    smiles1 = request.data.get('smiles1')
    smiles2 = request.data.get('smiles2')

    if not smiles1 or not smiles2:
        return Response({'error': 'Missing SMILES strings'}, status=400)

    try:
        # Generate image bytes
        img1_bytes = generate_mol_image(smiles1)
        img2_bytes = generate_mol_image(smiles2)

        # Base64 encode the image bytes to a string
        # .decode('utf-8') is used to convert the bytes result of b64encode into a string
        img1_b64 = base64.b64encode(img1_bytes).decode('utf-8') if img1_bytes else None
        img2_b64 = base64.b64encode(img2_bytes).decode('utf-8') if img2_bytes else None

        similarity = calculate_similarity(smiles1, smiles2)
        descriptors1 = calculate_molecular_descriptors(smiles1) or {}
        descriptors2 = calculate_molecular_descriptors(smiles2) or {}
        match = compare_functional_groups(smiles1, smiles2)

        score1 = descriptors1.get('LogP', 0) - descriptors1.get('TPSA', 0)
        score2 = descriptors2.get('LogP', 0) - descriptors2.get('TPSA', 0)

        prediction = ""
        if similarity is not None and similarity > 0.5 and match:
            prediction = "Molecule X likely binds better" if score1 > score2 else "Molecule Y likely binds better"
        elif similarity is not None and similarity <= 0.5:
            prediction = "Molecules have low similarity; binding prediction is less certain."
        else:
            prediction = "Could not determine binding prediction with available data."


        return Response({
            'smiles1': smiles1,
            'smiles2': smiles2,
            'similarity': similarity,
            'descriptors1': descriptors1,
            'descriptors2': descriptors2,
            'functional_group_match': match,
            'img1': img1_b64, 
            'img2': img2_b64, 
            'score1': score1,
            'score2': score2,
            'prediction': prediction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e), 'message': 'An internal server error occurred during prediction.'}, status=500)
    
# IHC insight view
# This view accepts an image file, processes it to count brown pixels, analyze shade distribution,
@api_view(['POST'])
@csrf_exempt
@permission_classes([IsAuthenticated])
def ihc_insight_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        random_filename = f"ihc_insight_{random.randint(1, 999999)}_{image_file.name}"
        file_path = default_storage.save(f'temp/{random_filename}', image_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            count = count_brown_pixels(full_path)
            distribution = brown_shade_distribution(full_path)
            mask_base64 = generate_brown_mask_image_base64(full_path)
            os.remove(full_path)

            return JsonResponse({
                'brown_pixel_count': count,
                'distribution': distribution,
                'mask_base64': mask_base64,
            })
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return HttpResponseBadRequest('Image not provided')

# Cell detection view
# This view accepts an image file, processes it to detect cells, and returns the results.
@api_view(['POST'])
@csrf_exempt
#@permission_classes([IsAuthenticated])
def cell_detection_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        random_filename = f"ihc_insight_{random.randint(1, 999999)}_{image_file.name}"
        file_path = default_storage.save(f'temp/{random_filename}', image_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            result = detect_cells(full_path)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return HttpResponseBadRequest('Image not provided')

# Heart disease prediction view
# This view accepts a JSON payload with heart disease risk factors and returns a prediction.
@api_view(['POST'])
def heart_disease_prediction_view(request):
    if request.method != 'POST':
        return Response({'error': 'Only POST method allowed'}, status=405)
    try:
        data = request.data
        age = int(data.get('age'))
        sex = int(data.get('sex'))
        cp = int(data.get('cp'))
        trestbps = int(data.get('trestbps'))
        chol = int(data.get('chol'))
        fbs = int(data.get('fbs'))
        restecg = int(data.get('restecg'))
        thalach = int(data.get('thalach'))
        exang = int(data.get('exang'))
        oldpeak = float(data.get('oldpeak'))
        slope = int(data.get('slope'))

        # Validate required fields
        required_fields = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]
        if any(field is None for field in required_fields):
            return Response({'error': 'All fields are required.'}, status=400)

        prediction, probability = predict_heart_disease(
            age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope
        )
        return Response({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)    
# Lung cancer prediction view
# This view accepts a JSON payload with lung cancer risk factors and returns a prediction.
@api_view(['POST'])
def lung_cancer_prediction_view(request):
    required_fields = [
        'AGE', 'GENDER', 'SMOKING', 'FINGER_DISCOLORATION', 'MENTAL_STRESS',
        'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS', 'ENERGY_LEVEL', 'IMMUNE_WEAKNESS',
        'BREATHING_ISSUE', 'ALCOHOL_CONSUMPTION', 'THROAT_DISCOMFORT', 'OXYGEN_SATURATION',
        'CHEST_TIGHTNESS', 'FAMILY_HISTORY', 'SMOKING_FAMILY_HISTORY', 'STRESS_IMMUNE'
    ]

    if request.method != 'POST':
        return Response({'error': 'Only POST method allowed'}, status=405)

    try:
        data = request.data

        # Check for missing fields
        for field in required_fields:
            if data.get(field) is None:
                return Response({'error': f'Missing field: {field}'}, status=400)

        # Prepare input as a dict for the model
        input_dict = {field: data.get(field) for field in required_fields}

        prediction_result = predict_pulmonary_disease(input_dict)

        return Response(prediction_result)

    except Exception as e:
        return Response({'error': f'Invalid input: {str(e)}'}, status=500)

# Lung cancer detection view
# This view accepts an image file, processes it to predict lung cancer, and returns the results
@api_view(['POST'])
@parser_classes([MultiPartParser])
def lung_cancer_detection_view(request):
    if 'image' not in request.FILES:
        return Response({'error': 'CT scan image is required.'}, status=400)

    try:
        image_file = request.FILES['image']
        image_bytes = image_file.read()

        prediction, probability, annotated_image = predict_lung_cancer_from_image(image_bytes)

        return Response({
            'prediction': prediction,
            'probability': round(probability, 2),
            'annotated_image': annotated_image
        })

    except Exception as e:
        return Response({'error': str(e)}, status=500)
    
# Research analyzer view
# This view accepts a PDF file, extracts text, summarizes it, and extracts keywords.
@api_view(['POST'])
@parser_classes([MultiPartParser])
@permission_classes([IsAuthenticated])
def research_analyzer_view(request):
    if 'file' not in request.FILES:
        return Response({'error': 'PDF file is required'}, status=400)

    file = request.FILES['file']
    filename = default_storage.save(f'temp/{file.name}', file)
    filepath = os.path.join(settings.MEDIA_ROOT, filename)

    try:
        from .research_analyzer import analyze_research_paper
        result = analyze_research_paper(filepath)
        return Response(result)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# Auto publish view
# This view handles multipart/form-data from the React AutoPublishTool, calls generate_report_from_form, and returns the generated PDF.
@api_view(['POST'])
@parser_classes([MultiPartParser])
@csrf_exempt
def auto_publish_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        metadata = json.loads(request.POST.get('metadata', '{}'))
        abstract = request.POST.get('abstract', '')
        sections = json.loads(request.POST.get('sections', '[]'))
        keywords = json.loads(request.POST.get('keywords', '[]'))
        images = request.FILES.getlist('images')

        print('--- auto_publish_view request values ---')
        print('metadata:', metadata)
        print('abstract:', abstract)
        print('sections:', sections)
        print('keywords:', keywords)
        print('images:', [img.name for img in images])

        tables = []
        charts = []

        # Parse uploaded tables
        for table_file in request.FILES.getlist('tables'):
            try:
                table_data = json.loads(table_file.read().decode('utf-8'))
                tables.append(table_data)
            except Exception as e:
                print('Error parsing table:', e)

        # Parse uploaded charts
        for chart_file in request.FILES.getlist('charts'):
            try:
                chart_data = json.loads(chart_file.read().decode('utf-8'))
                charts.append(chart_data)
            except Exception as e:
                print('Error parsing chart:', e)

        pdf_path = generate_report_from_form(metadata, abstract, sections, keywords, images, tables, charts)
        if os.path.exists(pdf_path):
            return FileResponse(open(pdf_path, 'rb'), as_attachment=True, filename='generated_report.pdf')
        else:
            return JsonResponse({'error': 'PDF generation failed'}, status=500)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


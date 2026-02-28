from django import forms
from authentication.models import User


class TextImportForm(forms.Form):
    """Form for importing text data files"""

    file = forms.FileField(
        label="Text File",
        help_text="Upload a .txt file with one text per line",
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.txt,text/plain'
        })
    )

    skip_empty_lines = forms.BooleanField(
        label="Skip Empty Lines",
        initial=True,
        required=False,
        help_text="Ignore empty lines in the file",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    auto_assign = forms.BooleanField(
        label="Auto-assign to Annotator",
        initial=False,
        required=False,
        help_text="Automatically assign imported texts to selected annotator",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    assigned_annotator = forms.ModelChoiceField(
        queryset=User.objects.filter(role='annotator', is_active=True),
        label="Assign to Annotator",
        required=False,
        empty_label="Select annotator...",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    auto_mcn_mapping = forms.BooleanField(
        label="Automatic MCN Mapping",
        initial=True,
        required=False,
        help_text="Automatically extract and map Indonesian medical terms to SNOMED-CT",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    def clean(self):
        cleaned_data = super().clean()
        auto_assign = cleaned_data.get('auto_assign')
        assigned_annotator = cleaned_data.get('assigned_annotator')

        if auto_assign and not assigned_annotator:
            raise forms.ValidationError("Please select an annotator when auto-assign is enabled.")

        return cleaned_data


class DocumentAssignmentForm(forms.Form):
    """Form for assigning documents to annotators"""

    annotator = forms.ModelChoiceField(
        queryset=User.objects.filter(role='annotator', is_active=True),
        label="Assign to Annotator",
        widget=forms.Select(attrs={'class': 'form-select'})
    )


class BulkAssignmentForm(forms.Form):
    """Form for bulk assignment of documents"""

    annotator = forms.ModelChoiceField(
        queryset=User.objects.filter(role='annotator', is_active=True),
        label="Assign to Annotator",
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    document_ids = forms.CharField(
        widget=forms.HiddenInput()
    )
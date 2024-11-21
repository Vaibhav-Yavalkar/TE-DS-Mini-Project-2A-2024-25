from django import forms

from core.models import Course

class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = '__all__'
        
        widgets = {
            'description': forms.Textarea(attrs={'style': "width:100%;"}),
        }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = (field.widget.attrs.get('class') or '') + ' regular-input'

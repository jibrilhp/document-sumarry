import unittest
from util import extract_filename_from_prompt, normalize_filename

class TestUtil(unittest.TestCase):
    def test_filename_extraction_positive_cases(self):
        """Test cases where a filename should be successfully extracted."""
        test_cases = {
            "Berdasarkan @statistic.csv, Berapa...": "statistic.csv",
            "Analyze data from @data_final.xlsx": "data_final.xlsx",
            "Can you analyze the file @report.csv?": "report.csv",
            "Please check @document.xlsx.": "document.xlsx",
            "Data is in @my-file_v2.json!": "my-file_v2.json",
            "Compare @data1.csv": "data1.csv",
        }

        for prompt, expected in test_cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(extract_filename_from_prompt(prompt), expected)

    def test_filename_extraction_negative_cases(self):
        """Test cases where no filename should be extracted."""
        test_cases = [
            "Berapa jumlah orang yang masuk ke Pelabuhan Tanjung Priok",
            "",
            "Please send the report to user@example.com",
            "The issue is with the @ symbol in the text.",
            "Hello world @",
            "Send email to user@example.com",
        ]

        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(extract_filename_from_prompt(prompt), "")

    def test_normalize_filename(self):
        """Test cases for filename normalization."""
        test_cases = {
            "test file.csv": "test_file.csv",
            "Test File.csv": "test_file.csv",
            "Another Document With Spaces.PDF": "another_document_with_spaces.pdf",
            "nospaces.txt": "nospaces.txt",
            "  leading_and_trailing_spaces  .doc  ": "_leading_and_trailing_spaces_.doc_",
        }

        for original, expected in test_cases.items():
            with self.subTest(original=original):
                self.assertEqual(normalize_filename(original), expected)

if __name__ == '__main__':
    unittest.main()
